from __future__ import annotations

import logging

from jevrag.retrieval import search

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

QUERY = "How many parameters does GPT-3 have?"


def main() -> None:
    hits = search(QUERY, top_k=5)

    print(f"\nQuery: {QUERY}\n")
    for rank, hit in enumerate(hits, start=1):
        print(
            f"#{rank}  score={hit.score:.4f}  "
            f"dense_rank={hit.dense_rank}  sparse_rank={hit.sparse_rank}  "
            f"{hit.source} p.{hit.page_number}"
        )
        print(f"    {hit.text[:200]}\n")


if __name__ == "__main__":
    main()
