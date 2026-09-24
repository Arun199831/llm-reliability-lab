from __future__ import annotations

import logging
import sys

from jevrag.pdf_loader import PdfLoadError, load_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: inspect_pdf.py <file.pdf>", file=sys.stderr)
        return 2

    try:
        pages = load_pdf(sys.argv[1])
    except PdfLoadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # Apple Rule: default before the loop.
    total = 0
    for p in pages:
        total += len(p.text)

    print(f"\n{pages[0].source}: {len(pages)} pages with text, {total:,} chars")
    for p in pages[:3]:
        paragraphs = p.text.count("\n\n") + 1
        print(f"  p.{p.page_number}: {len(p.text):,} chars, {paragraphs} paragraphs")

    print("\n--- page 1, first 600 chars ---")
    print(pages[0].text[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
