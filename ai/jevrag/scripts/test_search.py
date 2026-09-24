from __future__ import annotations

import logging

from jevrag.retrieval import search

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TEST_QUERIES = [
    "What are the two components that make up a RAG model?",
    "How does multi-head attention work?",
    "How many parameters does GPT-3 have?",
]


def main() -> None:
    for query in TEST_QUERIES:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print("=" * 70)

        hits = search(query, top_k=3)
        for rank, hit in enumerate(hits, start=1):
            print(
                f"\n  #{rank}  score={hit.score:.3f}  {hit.source} p.{hit.page_number}"
            )
            print(f"      {hit.text[:200]}")


if __name__ == "__main__":
    main()
