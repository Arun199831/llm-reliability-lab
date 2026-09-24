from __future__ import annotations

import logging

import litellm

from jevrag.config import Settings, get_settings
from jevrag.context import build_context
from jevrag.retrieval import SearchHit

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You answer questions strictly from the numbered context passages supplied.

Rules:
- Use only information present in the context. Never add outside knowledge.
- Cite every factual sentence with the passage number it came from, like [2].
- If the context does not fully support a claim, do not make that claim.
- Be concise. No preamble, no restating the question.
"""


class GenerationUnavailable(RuntimeError):
    """The LLM could not produce a draft. The caller decides what to do next
    -- this module never silently swallows a failed generation."""


async def generate_answer(
    question: str,
    hits: list[SearchHit],
    settings: Settings | None = None,
) -> str:
    s = settings or get_settings()
    context = build_context(hits)
    user_prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}"

    try:
        response = await litellm.acompletion(
            model=s.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            timeout=s.llm_timeout_s,
            num_retries=2,
        )
    except Exception as e:  # noqa: BLE001 - provider SDKs raise widely
        raise GenerationUnavailable(f"generation failed: {e}") from e

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise GenerationUnavailable("provider returned an empty completion")

    log.info("generation.complete question=%r chars=%d", question, len(answer))
    return answer
