from __future__ import annotations

import logging

from jevrag.bm25_index import build_bm25_index
from jevrag.chunking import chunk_pages
from jevrag.indexing import build_index
from jevrag.pdf_loader import PdfLoadError, load_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PDFS = ["data/attention.pdf", "data/rag.pdf", "data/gpt3.pdf"]


def main() -> int:
    all_chunks = []
    for path in PDFS:
        try:
            pages = load_pdf(path)
        except PdfLoadError as e:
            print(f"error loading {path}: {e}")
            return 1
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        print(f"{path}: {len(pages)} pages -> {len(chunks)} chunks")

    print(f"\ntotal chunks to index: {len(all_chunks)}")

    build_index(all_chunks)
    build_bm25_index(all_chunks)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
