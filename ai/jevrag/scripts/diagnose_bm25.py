"""Diagnostic: why doesn''t hybrid retrieval surface the "175 billion
parameters" fact? Shows, for every chunk that contains "175", exactly which
query tokens it shares, its real BM25 score, and its rank out of 444.
"""

from __future__ import annotations

from jevrag.bm25_index import load_bm25_index, tokenize

QUERY = "How many parameters does GPT-3 have?"


def main() -> None:
    bm25, chunks = load_bm25_index()
    query_tokens = set(tokenize(QUERY))
    scores = bm25.get_scores(tokenize(QUERY))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rank_of = {}
    rank = 1   # Apple Rule
    for i in ranked:
        rank_of[i] = rank
        rank += 1

    print(f"Query tokens: {sorted(query_tokens)}\n")
    print("Chunks containing \"175\" -- token overlap with the query, real "
          "BM25 score, rank out of 444:\n")

    for i, chunk in enumerate(chunks):
        if "175" in chunk.text:
            chunk_tokens = set(tokenize(chunk.text))
            overlap = query_tokens & chunk_tokens
            print(f"  [{i}] rank={rank_of[i]:3d}  score={scores[i]:.3f}  "
                  f"overlap={sorted(overlap)}")
            print(f"       {chunk.source} p.{chunk.page_number}: {chunk.text[:120]}")


if __name__ == "__main__":
    main()
