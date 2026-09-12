from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant answering questions about NLP papers.\n"
            "Answer ONLY using the provided context. Do not use outside knowledge.\n"
            "If the context does not contain the answer, say exactly: "
            "'The provided context does not contain this information.'\n\n"
            "Cite every factual claim with the chunk_id it came from, in square "
            "brackets, e.g. [attention.pdf_p8_c2]. Do not cite chunks you did not use.\n"
            "Be careful: chunks may mention similar terms from DIFFERENT papers or "
            "models. Only use chunks that are actually about what the question asks.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_context(documents: list[Document]) -> str:
    """Render retrieved documents into a citable context block.
    Each chunk is labelled with its chunk_id so the model can cite it."""
    if not documents:
        return "(no context retrieved)"

    blocks = []
    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", "?")
        blocks.append(
            f"[{chunk_id}] (source: {source}, page: {page})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def build_answer_chain(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Build the generation chain. temperature=0 for reproducibility —
    matters because RAGAS scores in Module 5 should measure the pipeline,
    not sampling randomness."""
    llm = ChatOpenAI(model=model, temperature=temperature)
    return ANSWER_PROMPT | llm


def answer_question(question: str, retriever, model: str = "gpt-4o-mini") -> dict:
    """Retrieve, generate, and return the answer alongside the exact
    context used — needed for citation checking and RAGAS evaluation."""
    documents = retriever.invoke(question)

    if not documents:
        return {
            "question": question,
            "answer": "The provided context does not contain this information.",
            "documents": [],
            "chunk_ids": [],
        }

    chain = build_answer_chain(model=model)

    try:
        response = chain.invoke(
            {
                "context": format_context(documents),
                "question": question,
            }
        )
    except Exception as e:
        raise RuntimeError(f"Generation failed for question '{question}': {e}")

    return {
        "question": question,
        "answer": response.content,
        "documents": documents,
        "chunk_ids": [d.metadata.get("chunk_id") for d in documents],
    }
