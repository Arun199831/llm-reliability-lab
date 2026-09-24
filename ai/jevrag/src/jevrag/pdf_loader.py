from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pymupdf
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

_MIN_PAGE_CHARS = 20  # below this a page is blank, a figure, or a scan
_TEXT_BLOCK = 0  # pymupdf block_type: 0 = text, 1 = image


class PdfLoadError(RuntimeError):
    """The file as a whole cannot be used. Per-page problems never raise."""


class Page(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: str
    page_number: int  # 1-based, matching what a reader sees
    text: str


def _clean_block(text: str) -> str:
    """Normalise one text block into one clean paragraph."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _page_text(page: pymupdf.Page) -> str:
    """Join a page's text blocks with blank lines, so each block becomes a
    paragraph that the chunker in step 4 can pack."""
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

    # Apple Rule: empty list before the loop starts.
    paragraphs = []
    for block in blocks:
        block_type = block[6]
        if block_type == _TEXT_BLOCK:
            raw_text = block[4]
            cleaned = _clean_block(raw_text)
            paragraphs.append(cleaned)

    non_empty_paragraphs = []
    for paragraph in paragraphs:
        if paragraph:
            non_empty_paragraphs.append(paragraph)

    return "\n\n".join(non_empty_paragraphs)


def load_pdf(path: str | Path) -> list[Page]:
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PdfLoadError(f"PDF not found: {pdf_path}")

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:  # noqa: BLE001
        raise PdfLoadError(f"cannot open {pdf_path.name}: {e}") from e

    pages: list[Page] = []
    skipped: list[int] = []

    try:
        if doc.needs_pass:
            raise PdfLoadError(f"{pdf_path.name} is password-protected")

        for index in range(doc.page_count):
            number = index + 1
            try:
                text = _page_text(doc.load_page(index))
            except Exception as e:
                log.warning(
                    "pdf.page_failed file=%s page=%d err=%s", pdf_path.name, number, e
                )
                skipped.append(number)
                continue

            if len(text) < _MIN_PAGE_CHARS:
                skipped.append(number)
                continue

            pages.append(
                Page(
                    doc_id=pdf_path.stem,
                    source=pdf_path.name,
                    page_number=number,
                    text=text,
                )
            )
    finally:
        doc.close()

    if skipped:
        log.warning("pdf.pages_skipped file=%s pages=%s", pdf_path.name, skipped)

    if not pages:
        raise PdfLoadError(
            f"no extractable text in {pdf_path.name}. If it is a scanned "
            "document, it needs OCR, which this loader does not do."
        )

    log.info("pdf.loaded file=%s pages=%d", pdf_path.name, len(pages))
    return pages
