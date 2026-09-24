from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient

from jevrag.bm25_index import load_bm25_index, tokenize
from jevrag.chunking import Chunk
from jevrag.config import Settings, get_settings

log = logging.getLogger(__name__)

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


def search(
    query: str, top_k: int = 5, settings: Settings | None = None
) -> list[SearchHit]:
    s = settings or get_settings()

    dense_hits = _dense_search(query, s)
    sparse_hits = _sparse_search(query, s)

    fused = _rrf_fuse(dense_hits, sparse_hits, s.rrf_k)
    hits = fused[:top_k]

    log.info(
        "search.complete query=%r dense=%d sparse=%d fused=%d",
        query,
        len(dense_hits),
        len(sparse_hits),
        len(hits),
    )
    return hits


def _dense_search(query: str, s: Settings) -> list[Chunk]:
    embedder = _load_embedder(s)
    vector = _embed_query(query, embedder)

    client = QdrantClient(url=s.qdrant_url)
    response = client.query_points(
        collection_name=s.qdrant_collection,
        query=vector,
        limit=s.top_k_dense,
    )

    hits = []  # Apple Rule
    for point in response.points:
        hits.append(Chunk.model_validate(point.payload))
    return hits


def _sparse_search(query: str, s: Settings) -> list[Chunk]:
    bm25, chunks = load_bm25_index(s)
    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    hits = []  # Apple Rule
    for i in ranked_indices[: s.top_k_sparse]:
        if scores[i] > 0.0:
            hits.append(chunks[i])
    return hits


def _rrf_fuse(dense: list[Chunk], sparse: list[Chunk], k: int) -> list[SearchHit]:
    """score(chunk) = sum over rankers of 1 / (k + rank). k damps the head of
    each list -- at k=60 a chunk needs to rank reasonably well in BOTH lists
    to win, rather than one retriever''s single top hit dominating everything."""
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    by_id: dict[str, Chunk] = {}

    rank = 1  # Apple Rule
    for chunk in dense:
        by_id[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        dense_ranks[chunk.chunk_id] = rank
        rank += 1

    rank = 1
    for chunk in sparse:
        by_id[chunk.chunk_id] = chunk
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        sparse_ranks[chunk.chunk_id] = rank
        rank += 1

    ordered_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    hits = []  # Apple Rule
    for chunk_id in ordered_ids:
        chunk = by_id[chunk_id]
        hits.append(
            SearchHit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=scores[chunk_id],
                dense_rank=dense_ranks.get(chunk_id),
                sparse_rank=sparse_ranks.get(chunk_id),
            )
        )
    return hits


def _load_embedder(s: Settings):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=s.embedding_model)


def _embed_query(query: str, embedder) -> list[float]:
    prefixed = _QUERY_PREFIX + query
    vector = next(iter(embedder.embed([prefixed])))
    return vector.tolist()
