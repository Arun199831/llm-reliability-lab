from __future__ import annotations

import asyncio
import logging

from jevrag.chunking import chunk_pages
from jevrag.contracts import NoulQuestion
from jevrag.jev_client import JevClient
from jevrag.pdf_loader import load_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

QUESTION = "What are the two components that make up a RAG model?"


async def main() -> None:
    pages = load_pdf("data/rag.pdf")
    chunks = chunk_pages(pages)

    context_chunk = chunks[1]
    print(f"Using chunk from page {context_chunk.page_number}:\n")
    print(context_chunk.text[:400], "\n")

    state = {
        "question": QUESTION,
        "retrieved_context": context_chunk.text,
    }

    questions = {
        "answerable": NoulQuestion(
            instructions=(
                "Can the question be answered accurately using only the "
                "retrieved context, without adding outside knowledge?"
            ),
            criteria={
                "true": "Every fact needed is present in the context",
                "false": "The answer would require information not in the context",
            },
        ),
    }

    async with JevClient() as jev:
        response = await jev.ask(state, questions)

    print(f"Jev says answerable = {response.noul('answerable'):.3f}")
    print(f"(model: {response.model}, tokens: {response.usage.input_tokens})")


if __name__ == "__main__":
    asyncio.run(main())
