# jevrag — RAG with a decision gate

A RAG pipeline over three ML papers (Attention Is All You Need, RAG, GPT-3) where,
before calling the LLM, a decision model checks whether the retrieved evidence can
actually answer the question. If not, the pipeline refuses and skips the LLM call.

The gate uses [Jev](https://typesafe.ai) (TypeSafe AI), a model that returns a
probability instead of text.

## Results: normal RAG vs Jev RAG

Same 16 questions, same retrieved chunks for both (10 answerable from the papers, 6 not).

|                              | Normal RAG | Jev RAG |
|------------------------------|-----------:|--------:|
| LLM calls                    | 16         | 8       |
| Unanswerable questions refused before the LLM | 0 / 6 | 6 / 6 |
| Cost per 1 lakh questions    | ₹1,352     | ₹1,157  |
| Avg time per question        | 1.72 s     | 1.12 s  |
| Tokens (all calls)           | 13,365     | 24,399  |

- Jev RAG used more tokens but cost less: the gate's tokens are much cheaper than LLM tokens.
- 2 answerable questions were refused. In both cases retrieval never returned the passage
  containing the answer, so the gate was right to refuse. Retrieval is the next thing to fix.
- Savings depend on how often questions get refused; this test set has 6 unanswerable questions by design.

Costs at 1 USD = ₹96.5. Retrieval time is excluded (identical for both).

## How it works

```
PDF → page extraction → paragraph-aware chunking (444 chunks)
    → Qdrant (bge-small embeddings) + BM25 → reciprocal rank fusion → top 3 chunks
    → Jev: "can this evidence answer the question?"  (threshold 0.6)
         ├─ yes → gpt-4o-mini writes a cited answer
         └─ no  → refuse, no LLM call
```

| File | Does |
|------|------|
| `pdf_loader.py` | PDF → cleaned pages (PyMuPDF) |
| `chunking.py` | Paragraph packing, overlap, stable hash IDs |
| `indexing.py` / `bm25_index.py` | Dense index in Qdrant, keyword index with BM25 |
| `retrieval.py` | Hybrid search with reciprocal rank fusion |
| `jev_client.py` | Jev API client with retries and backoff |
| `sufficiency.py` | The decision gate |
| `generation.py` | Cited answer via LiteLLM |
| `citations.py` | Checks which sentences carry citations |

## Run it

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, a TypeSafe API key and an OpenAI key.

```powershell
uv sync
# .env:  JEV_API_KEY=...   OPENAI_API_KEY=...

docker run -d --name qdrant -p 6333:6333 -v "${PWD}\qdrant_storage:/qdrant/storage" qdrant/qdrant

curl.exe -L -o data\attention.pdf https://arxiv.org/pdf/1706.03762
curl.exe -L -o data\rag.pdf       https://arxiv.org/pdf/2005.11401
curl.exe -L -o data\gpt3.pdf      https://arxiv.org/pdf/2005.14165

uv run python scripts/build_index.py     # build both indexes
uv run python scripts/test_pipeline.py   # two example questions end to end
uv run python scripts/compare.py         # normal RAG vs Jev RAG table
```

## What I learned

- **Dense retrieval finds the right topic, not always the right fact.** "How many parameters
  does GPT-3 have?" returned chunks about GPT-3's scale, none stating 175 billion.
- **Adding BM25 made that question worse.** BM25 weights words by how rare they are *in this
  corpus*. Most chunks come from the GPT-3 paper, so "GPT-3" and "parameters" are common here
  and get little weight.
- **Token count isn't cost.** Different models price tokens very differently.

## Limitations / next steps

- The 0.6 threshold is a starting value, not calibrated on labelled data yet.
- 16 questions is a small test set.
- Retrieval misses answerable facts; a reranker is the next step.
- Jev is an early-access model (released September 2026).