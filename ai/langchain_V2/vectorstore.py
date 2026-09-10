import time
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load the chunks.json produced by Module 1 ingestion"""
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks file found in {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError(f"{chunks_path} is empty - nothing to embed")
    return chunks


def chunks_to_document(chunks: list[dict]) -> tuple[list[Document], list[str]]:
    documents = []
    ids = []

    for chunk in chunks:
        documents.append(
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
        )
        ids.append(chunk["metadata"]["chunk_id"])

    return documents, ids


def build_vectorstore(
    chunks_path: Path,
    persist_dir: Path,
    collection_name: str = "rag_papers",
    embedding_model: str = "text-embedding-3-small",
    max_retries: int = 3,
) -> Chroma:
    """Load chunks, embed them, and store them in a persistent Chroma collection.
    Idempotent: chunk_id is used as the Chroma document ID, and existing IDs
    are deleted before insert — so re-running this never duplicates chunks"""

    chunks = load_chunks(chunks_path)
    documents, ids = chunks_to_document(chunks)

    embeddings = OpenAIEmbeddings(model=embedding_model)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    try:
        vector_store.delete(ids=ids)
    except Exception as e:
        print(f"Delete step skipped (expected on first run): {e}")

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            vector_store.add_documents(documents=documents, ids=ids)
            print(
                f"Embedded and stored {len(documents)} chunks in '{collection_name}'."
            )
            return vector_store
        except Exception as e:
            last_error = e
            print(f"Embedding attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Failed to embed and store chunks after {max_retries} attempts: {last_error}"
    )
