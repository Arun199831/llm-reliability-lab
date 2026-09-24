"""Step 9 -- sparse (BM25) indexing and search.

Dense embeddings match meaning and miss literals; BM25 matches literals and
misses paraphrase. That only holds if the tokenizer preserves literals
correctly -- splitting "GPT-3" into "gpt" and "3" destroys the one thing
that made it useful to search on, because the bare token "3" is common
enough in any technical paper (section numbers, table numbers, figure
numbers) to match chunks that have nothing to do with GPT-3 at all. Keeping
hyphenated technical terms as one token is what makes this retriever''s
literal-matching actually work.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from jevrag.chunking import Chunk
from jevrag.config import Settings, get_settings

log = logging.getLogger(__name__)

# Matches "gpt-3", "175b", "state-of-the-art" as single tokens, while still
# splitting on spaces and other punctuation. The hyphen is kept INSIDE a
# token, never used to split one -- that is the whole fix.
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def build_bm25_index(chunks: list[Chunk], settings: Settings | None = None) -> None:
    s = settings or get_settings()

    if not chunks:
        log.warning("bm25.no_chunks -- nothing to index")
        return

    corpus = []   # Apple Rule
    for chunk in chunks:
        corpus.append(tokenize(chunk.text))

    bm25 = BM25Okapi(corpus)

    path = Path(s.bm25_index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    log.info("bm25.complete chunks=%d path=%s", len(chunks), path)


def load_bm25_index(settings: Settings | None = None) -> tuple[BM25Okapi, list[Chunk]]:
    s = settings or get_settings()
    path = Path(s.bm25_index_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"BM25 index not found at {path} -- run scripts/build_index.py first"
        )

    with path.open("rb") as f:
        blob = pickle.load(f)

    return blob["bm25"], blob["chunks"]


def tokenize(text: str) -> list[str]:
    """No stemming, no stop-word removal -- BM25''s whole contribution to
    hybrid retrieval is exact-term matching, and stemming would blur exactly
    the technical literals this retriever exists to catch. Hyphenated
    compounds ARE kept whole, though: "gpt-3" as one token IS the literal;
    splitting it on the hyphen was the actual bug, not a feature of keeping
    things simple."""
    return _TOKEN.findall(text.lower())
