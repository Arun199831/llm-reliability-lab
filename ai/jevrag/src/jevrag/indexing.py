from __future__ import annotations

import logging
import time

from qdrant_client import QdrantClient, models

from jevrag.chunking import Chunk
from jevrag.config import Settings, get_settings

log = logging.getLogger(__name__)

_UPSERT_BATCH = 128
_MAX_ATTEMPTS = 4


def build_index(chunks: list[Chunk], settings: Settings | None = None) -> None:
    s = settings or get_settings()

    if not chunks:
        log.warning("index.no_chunks -- nothing to index")
        return

    embedder = _load_embedder(s)
    client = QdrantClient(url=s.qdrant_url)

    _recreate_collection(client, s)
    points = _embed_chunks(chunks, embedder)
    _upsert_points(client, s, points)

    log.info("index.complete chunks=%d collection=%s", len(chunks), s.qdrant_collection)


def _load_embedder(s: Settings):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=s.embedding_model)


def _recreate_collection(client: QdrantClient, s: Settings) -> None:
    """Drop and rebuild. Correct for a reproducible, from-source corpus like
    this one -- a production system would upsert by stable chunk_id and
    reconcile deletions separately instead of wiping the collection each run."""
    client.recreate_collection(
        collection_name=s.qdrant_collection,
        vectors_config=models.VectorParams(
            size=s.embedding_dim,
            distance=models.Distance.COSINE,
        ),
    )


def _embed_chunks(chunks: list[Chunk], embedder) -> list[models.PointStruct]:
    # bge-small is trained asymmetrically: passages are embedded bare here,
    # only the query side gets a prefix at retrieval time. See retrieval.py.
    texts = [c.text for c in chunks]
    vectors = list(embedder.embed(texts))

    points: list[models.PointStruct] = []
    index = 0
    for chunk, vector in zip(chunks, vectors, strict=True):
        # Qdrant's own point id is just a sequential int -- chunk.chunk_id
        # (the stable hash) rides along in the payload and is what citations
        # actually reference downstream.
        points.append(
            models.PointStruct(
                id=index,
                vector=vector.tolist(),
                payload=chunk.model_dump(),
            )
        )
        index += 1
    return points


def _upsert_points(
    client: QdrantClient, s: Settings, points: list[models.PointStruct]
) -> None:
    start = 0  # Apple Rule
    while start < len(points):
        batch = points[start : start + _UPSERT_BATCH]
        _upsert_batch_with_retry(client, s, batch)
        start += _UPSERT_BATCH


def _upsert_batch_with_retry(
    client: QdrantClient, s: Settings, batch: list[models.PointStruct]
) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            client.upsert(collection_name=s.qdrant_collection, points=batch, wait=True)
            return
        except Exception as e:  # noqa: BLE001 - qdrant-client raises several types
            last_exc = e
            log.warning(
                "index.upsert_failed attempt=%d/%d err=%s", attempt, _MAX_ATTEMPTS, e
            )
            if attempt == _MAX_ATTEMPTS:
                break
            time.sleep(0.5 * attempt)
    raise RuntimeError(
        f"failed to upsert batch after {_MAX_ATTEMPTS} attempts"
    ) from last_exc
