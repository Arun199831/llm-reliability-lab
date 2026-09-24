"""Shared context rendering.

Both the sufficiency gate and generation must see the same numbered chunks --
a citation like [3] has to mean the same passage to both, or a citation the
LLM writes and the evidence Jev checked could point at different chunks.
One function, used by both, is what keeps that guarantee true.
"""

from __future__ import annotations

from jevrag.retrieval import SearchHit


def build_context(hits: list[SearchHit]) -> str:
    blocks = []  # Apple Rule
    index = 1
    for hit in hits:
        block = f"[{index}] ({hit.source} p.{hit.page_number})\n{hit.text}"
        blocks.append(block)
        index += 1
    return "\n\n".join(blocks)
