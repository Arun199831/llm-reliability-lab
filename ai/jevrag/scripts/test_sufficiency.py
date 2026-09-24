"""Step 7 check: the sufficiency gate on two real queries.

One case where retrieval found the actual fact (multi-head attention), one
where it didn''t (step 6 showed the GPT-3 parameter-count query pulling
chunks that discuss GPT-3''s scale without ever stating the number).
"""

from __future__ import annotations

import asyncio
import logging

from jevrag.retrieval import search
from jevrag.sufficiency import check_sufficiency

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TEST_CASES = [
    "How does multi-head attention work?",
    "How many parameters does GPT-3 have?",
]


async def run_one(question: str) -> None:
    hits = search(question, top_k=3)

    print(f"\n{'=' * 70}")
    print(f"Question: {question}")
    print("=" * 70)
    for hit in hits:
        print(f"  {hit.source} p.{hit.page_number}: {hit.text[:150]}")

    result = await check_sufficiency(question, hits)

    print(f"\n  sufficient = {result.sufficient}")
    print(f"  score      = {result.answerable_score}")
    print(f"  reason     = {result.reason}")


async def main() -> None:
    for question in TEST_CASES:
        await run_one(question)


if __name__ == "__main__":
    asyncio.run(main())
