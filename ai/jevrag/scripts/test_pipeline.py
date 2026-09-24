from __future__ import annotations

import asyncio
import logging

from jevrag.generation import GenerationUnavailable, generate_answer
from jevrag.retrieval import search
from jevrag.sufficiency import check_sufficiency

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TEST_CASES = [
    "How does multi-head attention work?",
    "How many parameters does GPT-3 have?",
]


async def run_one(question: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"Question: {question}")
    print("=" * 70)

    hits = search(question, top_k=3)
    result = await check_sufficiency(question, hits)

    print(f"  sufficiency: {result.sufficient}  ({result.reason})")

    if not result.sufficient:
        print("  -> refusing. No LLM call made.")
        return

    try:
        answer = await generate_answer(question, hits)
    except GenerationUnavailable as e:
        print(f"  -> generation failed: {e}")
        return

    print(f"\n  Answer:\n  {answer}")


async def main() -> None:
    for question in TEST_CASES:
        await run_one(question)


if __name__ == "__main__":
    asyncio.run(main())
