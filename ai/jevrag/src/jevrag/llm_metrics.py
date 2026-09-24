from __future__ import annotations

import logging

import litellm

log = logging.getLogger(__name__)


def usage_tokens(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


def cost_usd(response) -> float:
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception as e:  # noqa: BLE001 - unknown model prices raise
        log.warning("llm.cost_unavailable err=%s", e)
        return 0.0
