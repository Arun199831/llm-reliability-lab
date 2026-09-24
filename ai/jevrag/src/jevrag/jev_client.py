from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Mapping

import httpx
from pydantic import ValidationError

from jevrag.config import Settings, get_settings
from jevrag.contracts import JevResponse, NoulQuestion

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}


class JevError(Exception):
    """Base. Catch this to apply your own fallback policy."""


class JevConfigError(JevError):
    """401/403 -- bad or missing key. Never retried."""


class JevRequestError(JevError):
    """400/422 -- our payload is wrong. Never retried."""


class JevUnavailableError(JevError):
    """Retries exhausted, or the network never came back."""


class JevContractError(JevError):
    """Jev answered, but not in the shape we declared."""


class JevClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._http = httpx.AsyncClient(
            base_url=self._s.jev_base_url,
            timeout=httpx.Timeout(
                self._s.jev_timeout_s, connect=self._s.jev_connect_timeout_s
            ),
            headers={
                "Authorization": f"Bearer {self._s.jev_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )

    async def __aenter__(self) -> "JevClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._http.aclose()

    async def ask(
        self, state: str | Mapping[str, Any], questions: Mapping[str, NoulQuestion]
    ) -> JevResponse:
        if not questions:
            raise JevRequestError("at least one question is required")

        payload = {
            "model": self._s.jev_model,
            "state": state,
            "questions": {
                name: q.model_dump(exclude_none=True) for name, q in questions.items()
            },
        }

        raw = await self._post_with_retry("/systemone", payload)

        try:
            return JevResponse.model_validate(raw)
        except ValidationError as e:
            raise JevContractError(f"unexpected response shape: {e}") from e

    async def _post_with_retry(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        response: httpx.Response | None = None  # Apple Rule

        for attempt in range(1, self._s.jev_max_attempts + 1):
            try:
                response = await self._http.post(path, json=payload)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exc = e
                log.warning(
                    "jev.transport_error attempt=%d/%d err=%s",
                    attempt,
                    self._s.jev_max_attempts,
                    e,
                )
                if attempt == self._s.jev_max_attempts:
                    break
                await asyncio.sleep(self._backoff(attempt))
                continue

            status = response.status_code
            if status == 200:
                return response.json()
            if status in (401, 403):
                raise JevConfigError(f"auth failed ({status}) -- check JEV_API_KEY")
            if status in (400, 422):
                raise JevRequestError(
                    f"invalid request ({status}): {response.text[:400]}"
                )
            if status in _RETRYABLE_STATUS:
                last_exc = JevUnavailableError(f"HTTP {status}")
                if attempt == self._s.jev_max_attempts:
                    break
                await asyncio.sleep(self._delay_for(response, attempt))
                continue
            raise JevUnavailableError(
                f"unexpected status {status}: {response.text[:400]}"
            )

        raise JevUnavailableError(
            f"exhausted {self._s.jev_max_attempts} attempts"
        ) from last_exc

    def _delay_for(self, response: httpx.Response, attempt: int) -> float:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return min(float(header), self._s.jev_backoff_max_s)
            except ValueError:
                pass
        return self._backoff(attempt)

    def _backoff(self, attempt: int) -> float:
        ceiling = min(
            self._s.jev_backoff_max_s,
            self._s.jev_backoff_base_s * (2 ** (attempt - 1)),
        )
        return random.uniform(0.0, ceiling)
