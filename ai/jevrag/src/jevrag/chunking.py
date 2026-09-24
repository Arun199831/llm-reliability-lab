from __future__ import annotations

import hashlib
import logging

from pydantic import BaseModel, ConfigDict

from jevrag.pdf_loader import Page

log = logging.getLogger(__name__)

_TARGET_CHARS = 1200
_OVERLAP_CHARS = 150
_MIN_CHUNK_CHARS = 40


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str


def chunk_pages(
    pages: list[Page],
    target_chars: int = _TARGET_CHARS,
    overlap_chars: int = _OVERLAP_CHARS,
) -> list[Chunk]:
    # Apple Rule: empty list before the loop starts.
    all_chunks: list[Chunk] = []

    for page in pages:
        page_chunks = _chunk_page(page, target_chars, overlap_chars)
        all_chunks.extend(page_chunks)

    log.info("chunk.complete pages=%d chunks=%d", len(pages), len(all_chunks))
    return all_chunks


def _chunk_page(page: Page, target_chars: int, overlap_chars: int) -> list[Chunk]:
    paragraphs = page.text.split("\n\n")
    pieces = _pack_paragraphs(paragraphs, target_chars, overlap_chars)

    chunks: list[Chunk] = []
    index = 0
    for piece in pieces:
        piece = piece.strip()
        if len(piece) < _MIN_CHUNK_CHARS:
            continue
        chunks.append(
            Chunk(
                chunk_id=_stable_chunk_id(page.doc_id, page.page_number, index, piece),
                doc_id=page.doc_id,
                source=page.source,
                page_number=page.page_number,
                chunk_index=index,
                text=piece,
            )
        )
        index += 1
    return chunks


def _pack_paragraphs(
    paragraphs: list[str], target_chars: int, overlap_chars: int
) -> list[str]:
    """Greedily pack paragraphs up to target_chars, carrying overlap_chars of
    tail into the next chunk. A paragraph bigger than target_chars on its own
    is handed off to _hard_split, since packing can't help something that
    doesn't fit regardless of what it's packed alongside."""
    pieces: list[str] = []
    buf = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > target_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_hard_split(paragraph, target_chars, overlap_chars))
            continue

        would_overflow = buf and len(buf) + len(paragraph) + 2 > target_chars
        if would_overflow:
            pieces.append(buf)
            buf = _start_new_chunk(buf, paragraph, target_chars, overlap_chars)
        elif buf:
            buf = buf + "\n\n" + paragraph
        else:
            buf = paragraph

    if buf.strip():
        pieces.append(buf)

    return pieces


def _start_new_chunk(
    previous_buf: str, paragraph: str, target_chars: int, overlap_chars: int
) -> str:
    """Seed a new chunk with an overlap tail -- but only if the tail actually
    fits alongside the paragraph. Without this check, overlap + a near-target
    paragraph can exceed target_chars before any further packing even runs."""
    if not overlap_chars:
        return paragraph
    tail = previous_buf[-overlap_chars:]
    candidate = tail + "\n\n" + paragraph
    if len(candidate) <= target_chars:
        return candidate
    return paragraph


def _hard_split(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Last resort for a single paragraph longer than the target chunk size.
    Ignores paragraph structure entirely -- this is a fallback, not the
    normal path."""
    step = max(1, target_chars - overlap_chars)
    pieces: list[str] = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + target_chars])
        start += step
    return pieces


def _stable_chunk_id(doc_id: str, page_number: int, chunk_index: int, text: str) -> str:
    """Hash-based, not a random UUID -- re-chunking an unchanged page must
    produce the same ids, or a citation from a previous run stops resolving."""
    h = hashlib.sha256()
    h.update(doc_id.encode())
    h.update(str(page_number).encode())
    h.update(str(chunk_index).encode())
    h.update(text.encode())
    return h.hexdigest()[:24]
