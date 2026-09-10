from pathlib import Path
from dotenv import load_dotenv
from vectorstore import build_vectorstore

load_dotenv()  # loads OPENAI_API_KEY from .env

CHUNKS_PATH = Path(__file__).parent / "processed" / "chunks.json"
PERSIST_DIR = Path(__file__).parent / "chroma_db"

if __name__ == "__main__":
    build_vectorstore(chunks_path=CHUNKS_PATH, persist_dir=PERSIST_DIR)
