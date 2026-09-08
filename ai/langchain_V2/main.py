from ingestion import ingest_pdf


from pathlib import Path
from ingestion import ingest_pdf

DATA_DIR = Path(__file__).parent / "data"


from pathlib import Path
from ingestion import ingest_all_pdfs, save_chunks

DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "processed" / "chunks.json"

if __name__ == "__main__":
    chunks = ingest_all_pdfs(DATA_DIR, ingestion_run_id="2026-09-08_v1")
    print(f"\nTotal chunks across all PDFs: {len(chunks)}")
    save_chunks(chunks, OUT_PATH)
