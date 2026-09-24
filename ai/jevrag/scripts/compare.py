from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import litellm

from jevrag.config import get_settings
from jevrag.context import build_context
from jevrag.generation import _SYSTEM_PROMPT  # same prompt as the real pipeline
from jevrag.jev_client import JevClient
from jevrag.retrieval import search
from jevrag.sufficiency import check_sufficiency

QUESTIONS_FILE = Path("eval/questions.jsonl")
USD_TO_INR = float(os.environ.get("USD_TO_INR", "96.5"))
JEV_USD_PER_TOKEN = 0.042 / 1_000_000


async def main() -> None:
    questions = load_questions()
    if not questions:
        print(f"no questions found in {QUESTIONS_FILE}")
        return

    normal = new_totals()
    jev_rag = new_totals()

    async with JevClient() as jev:
        for question in questions:
            hits = search(question, top_k=3)  # same chunks for both

            # Normal RAG: always answer
            await call_llm(question, hits, normal)

            # Jev RAG: gate first
            start = time.perf_counter()
            gate = await check_sufficiency(question, hits, jev=jev)
            jev_rag["time_ms"] += ms_since(start)
            jev_rag["tokens"] += gate.input_tokens
            jev_rag["cost_inr"] += gate.input_tokens * JEV_USD_PER_TOKEN * USD_TO_INR

            if gate.sufficient:
                await call_llm(question, hits, jev_rag)
                decision = "answered"
            else:
                jev_rag["refused"] += 1
                decision = "REFUSED"

            print(f"{decision:<9} {gate.reason:<32} {question[:55]}")

    print_table(len(questions), normal, jev_rag)


async def call_llm(question: str, hits: list, totals: dict) -> None:
    s = get_settings()
    user_prompt = f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {question}"

    start = time.perf_counter()
    try:
        response = await litellm.acompletion(
            model=s.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            timeout=s.llm_timeout_s,
            num_retries=2,
        )
    except Exception as e:  # noqa: BLE001
        print(f"    LLM call failed: {e}")
        return

    totals["time_ms"] += ms_since(start)
    totals["llm_calls"] += 1
    totals["tokens"] += response.usage.total_tokens
    try:
        totals["cost_inr"] += (
            litellm.completion_cost(completion_response=response) * USD_TO_INR
        )
    except Exception:  # noqa: BLE001 - unknown model price
        print("    (no price available for this model)")


def print_table(n: int, normal: dict, jev_rag: dict) -> None:
    rows = [
        ("questions", n, n),
        ("LLM calls", normal["llm_calls"], jev_rag["llm_calls"]),
        ("refused before LLM", normal["refused"], jev_rag["refused"]),
        ("tokens (all calls)", normal["tokens"], jev_rag["tokens"]),
        ("cost (INR)", f"{normal['cost_inr']:.4f}", f"{jev_rag['cost_inr']:.4f}"),
        (
            "cost per 1 lakh Qs (INR)",
            f"{normal['cost_inr'] / n * 100_000:,.0f}",
            f"{jev_rag['cost_inr'] / n * 100_000:,.0f}",
        ),
        (
            "avg time per question (ms)",
            f"{normal['time_ms'] / n:.0f}",
            f"{jev_rag['time_ms'] / n:.0f}",
        ),
    ]
    print(f"\n{'':<28}{'normal RAG':>12}{'Jev RAG':>12}")
    print("-" * 52)
    for label, a, b in rows:
        print(f"{label:<28}{str(a):>12}{str(b):>12}")
    print(f"\n(1 USD = {USD_TO_INR} INR; retrieval time not included -- same for both)")


def new_totals() -> dict:
    return {"llm_calls": 0, "refused": 0, "tokens": 0, "cost_inr": 0.0, "time_ms": 0.0}


def load_questions() -> list[str]:
    questions = []  # Apple Rule
    if not QUESTIONS_FILE.is_file():
        return questions
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])
    return questions


def ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


if __name__ == "__main__":
    asyncio.run(main())
