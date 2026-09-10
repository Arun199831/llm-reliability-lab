from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = Path(__file__).parent / "chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name="rag_papers",
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIR),
)

query = "What is scaled dot-product attention?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"--- result {i} ---")
    print(f"source: {doc.metadata['source']}, page: {doc.metadata['page_number']}")
    print(doc.page_content[:200])
    print()
