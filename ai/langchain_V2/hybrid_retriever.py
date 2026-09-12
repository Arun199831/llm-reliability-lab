from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain_community.retrievers import EnsembleRetriever

from vectorstore import load_chunks, chunks_to_document


def build_bm25_retriever(chunks_path: Path, k: int = 5) -> BM25Retriever:
    """Build a BM25 (sparse/keyword) retriever from the full chunk corpus.
    No persistence: BM25's IDF statistics are corpus-wide and there's no
    embedding cost, so rebuilding fresh every run is correct, not a shortcut."""
    chunks = load_chunks(chunks_path)
    documents, _ = chunks_to_document(chunks)

    if not documents:
        raise ValueError("No documents available to build BM25 index from.")

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def build_dense_retriever(
    persist_dir: Path,
    collection_name: str = "rag_papers",
    embedding_model: str = "text-embedding-3-small",
    k: int = 5,
):
    """Load the already-persisted Chroma collection from Module 2 and wrap
    it as a retriever. Does not re-embed anything."""
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No Chroma index found at {persist_dir}. Run build_index.py first."
        )

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(
    chunks_path: Path,
    persist_dir: Path,
    collection_name: str = "rag_papers",
    embedding_model: str = "text-embedding-3-small",
    k: int = 5,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> EnsembleRetriever:
    """Combine BM25 (sparse) and Chroma (dense) into one retriever.
    Both run independently against the same query; results are merged
    via Reciprocal Rank Fusion, weighted by dense_weight/sparse_weight."""
    bm25_retriever = build_bm25_retriever(chunks_path, k=k)
    dense_retriever = build_dense_retriever(
        persist_dir, collection_name, embedding_model, k=k
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[sparse_weight, dense_weight],
    )
