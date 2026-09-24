from __future__ import annotations

import logging
import sys

from jevrag.chunking import chunk_pages
from jevrag.pdf_loader import PdfLoadError, load_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: inspect_chunks.py <file.pdf>", file=sys.stderr)
        return 2

    try:
        pages = load_pdf(sys.argv[1])
    except PdfLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    chunks = chunk_pages(pages)

    sizes = []
    for c in chunks:
        sizes.append(len(c.text))

    total_chars = 0
    for size in sizes:
        total_chars += size
    avg_chars = total_chars / len(sizes) if sizes else 0

    print(f"\n{pages[0].source}: {len(pages)} pages -> {len(chunks)} chunks")
    print(f"  chunk size: min={min(sizes)}  avg={avg_chars:.0f}  max={max(sizes)}")

    print("\n--- first 2 chunks ---")
    for c in chunks[:2]:
        print(
            f"\n[{c.chunk_id}] page {c.page_number}, chunk {c.chunk_index} "
            f"({len(c.text)} chars)"
        )
        print(c.text[:300])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
