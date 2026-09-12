from pathlib import Path
from dotenv import load_dotenv
from hybrid_retriever import build_hybrid_retriever

load_dotenv()

CHUNKS_PATH = Path(__file__).parent / "processed" / "chunks.json"
PERSIST_DIR = Path(__file__).parent / "chroma_db"

retriever = build_hybrid_retriever(CHUNKS_PATH, PERSIST_DIR)

query = "What BLEU score did the Transformer big model achieve on WMT 2014 English-to-German?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"--- result {i} ---")
    print(f"chunk_id: {doc.metadata['chunk_id']}")
    print(f"source: {doc.metadata['source']}")
    print(f"page: {doc.metadata['page_number']}")
    print(f"ingestion_run_id: {doc.metadata['ingestion_run_id']}")
    print()
