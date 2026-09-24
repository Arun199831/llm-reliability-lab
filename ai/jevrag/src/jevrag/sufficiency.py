from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from jevrag.config import Settings, get_settings
from jevrag.context import build_context
from jevrag.contracts import NoulQuestion
from jevrag.jev_client import JevClient, JevError
from jevrag.retrieval import SearchHit

log = logging.getLogger(__name__)

_QUESTION_KEY = "answerable"
_INSTRUCTIONS = (
    "Can the question be answered accurately using only the retrieved "
    "context, without adding outside knowledge?"
)
_CRITERIA = {
    "true": "Every fact needed is present in the context",
    "false": "The answer would require information not in the context",
}


class SufficiencyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    sufficient: bool
    answerable_score: float | None
    threshold: float
    degraded: bool
    reason: str
    input_tokens: int = 0


async def check_sufficiency(
    question: str,
    hits: list[SearchHit],
    settings: Settings | None = None,
    jev: JevClient | None = None,
) -> SufficiencyResult:
    """Pass a shared `jev` client to reuse one connection across many calls;
    without one, a client is created and closed for this call only."""
    s = settings or get_settings()

    if not hits:
        return SufficiencyResult(
            sufficient=False,
            answerable_score=None,
            threshold=s.sufficiency_threshold,
            degraded=False,
            reason="retrieval returned no chunks",
        )

    context = build_context(hits)
    state = {"question": question, "retrieved_context": context}
    questions = {
        _QUESTION_KEY: NoulQuestion(instructions=_INSTRUCTIONS, criteria=_CRITERIA),
    }

    try:
        if jev is not None:
            response = await jev.ask(state, questions)
        else:
            async with JevClient(s) as own_client:
                response = await own_client.ask(state, questions)
    except JevError as e:
        log.warning("sufficiency.degraded err=%s", e)
        return SufficiencyResult(
            sufficient=False,
            answerable_score=None,
            threshold=s.sufficiency_threshold,
            degraded=True,
            reason=f"jev unavailable: {e}",
        )

    score = response.noul(_QUESTION_KEY)
    return SufficiencyResult(
        sufficient=score >= s.sufficiency_threshold,
        answerable_score=score,
        threshold=s.sufficiency_threshold,
        degraded=False,
        reason=f"answerable={score:.3f} threshold={s.sufficiency_threshold:.2f}",
        input_tokens=response.usage.input_tokens,
    )
