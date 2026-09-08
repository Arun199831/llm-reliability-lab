from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import json


def ingest_pdf(pdf_path: str, ingestion_run_id: str) -> list[dict]:
    """Extract text from a PDF page by page, chunk each page independently,
    and return a list of dicts: {"text": ..., "metadata": {...}}"""

    source = Path(pdf_path).name

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)

    all_chunks = []

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"failed to open {pdf_path}: {e}")

    import json

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        page_text = page.get_text()

        if not page_text.strip():
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            all_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": f"{source}_p{page_number}_c{chunk_index}",
                        "source": source,
                        "page_number": page_number,
                        "ingestion_run_id": ingestion_run_id,
                    },
                }
            )

    doc.close()
    return all_chunks


def ingest_all_pdfs(data_dir: Path, ingestion_run_id: str) -> list[dict]:
    """Run ingest_pdf across every PDF in data_dir, return one combined list."""
    all_chunks = []
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    for pdf_path in pdf_files:
        print(f"Ingesting {pdf_path.name}...")
        chunks = ingest_pdf(str(pdf_path), ingestion_run_id)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    return all_chunks


def save_chunks(chunks: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {out_path}")
