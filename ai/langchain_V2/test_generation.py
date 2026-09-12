from pathlib import Path
from dotenv import load_dotenv
from hybrid_retriever import build_hybrid_retriever
from generation import answer_question

load_dotenv()

CHUNKS_PATH = Path(__file__).parent / "processed" / "chunks.json"
PERSIST_DIR = Path(__file__).parent / "chroma_db"

retriever = build_hybrid_retriever(CHUNKS_PATH, PERSIST_DIR)

questions = [
    "What BLEU score did the Transformer big model achieve on WMT 2014 English-to-German?",
    "What is the capital of France?",  # grounding test - not in any paper
]

for q in questions:
    result = answer_question(q, retriever)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Retrieved chunk_ids: {result['chunk_ids']}")
    print("-" * 70)
