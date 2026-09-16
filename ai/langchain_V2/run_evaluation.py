import json
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from hybrid_retriever import build_hybrid_retriever
from generation import answer_question
from ragas.run_config import RunConfig

load_dotenv()

BASE_DIR = Path(__file__).parent
CHUNKS_PATH = BASE_DIR / "processed" / "chunks.json"
PERSIST_DIR = BASE_DIR / "chroma_db"
GOLDEN_SET_PATH = BASE_DIR / "evaluation" / "golden_dataset.json"
RESULTS_PATH = BASE_DIR / "evaluation" / "results.json"

# Separate from the generation model on purpose: a stronger judge grades
# a cheaper answerer. Change one without disturbing the other.
JUDGE_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o-mini"


def load_golden_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No golden set found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    if not golden_set:
        raise ValueError(f"{path} is empty - nothing to evaluate")
    return golden_set


def build_eval_dataset(golden_set: list[dict], retriever) -> Dataset:
    """Run the full RAG pipeline over every golden question and assemble
    the four columns RAGAS needs: question, answer, contexts, ground_truth."""
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, item in enumerate(golden_set, start=1):
        question = item["question"]
        print(f"[{i}/{len(golden_set)}] {question[:60]}...")

        try:
            result = answer_question(question, retriever, model=GENERATION_MODEL)
        except Exception as e:
            print(f"  Skipped - pipeline failed: {e}")
            continue

        rows["question"].append(question)
        rows["answer"].append(result["answer"])
        rows["contexts"].append([d.page_content for d in result["documents"]])
        rows["ground_truth"].append(item["ground_truth"])

    if not rows["question"]:
        raise RuntimeError("No questions completed successfully - nothing to evaluate.")

    return Dataset.from_dict(rows)


def run_evaluation():
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    retriever = build_hybrid_retriever(CHUNKS_PATH, PERSIST_DIR)

    print(f"Running pipeline over {len(golden_set)} questions...\n")
    dataset = build_eval_dataset(golden_set, retriever)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0.0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    print(f"\nScoring with judge model {JUDGE_MODEL}...")
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=4, timeout=180, max_retries=5),
    )

    print("\n=== RAGAS scores ===")
    print(scores)

    df = scores.to_pandas()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(RESULTS_PATH, orient="records", indent=2)
    print(f"\nPer-question results saved to {RESULTS_PATH}")

    return scores, df


if __name__ == "__main__":
    run_evaluation()
