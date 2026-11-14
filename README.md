# NEET Doubt Solver — RAG over study notes

Students ask a NEET doubt in plain English and get a short, exam-focused answer **grounded in the study notes**, with numbered citations to the exact chapter and section. Questions the notes don't cover are refused instead of answered from guesswork.

Built with **Python, FastAPI, Google Gemini embeddings, a FAISS vector index and an LLM (Gemini or Claude)**.

```
notes (.md/.pdf) ─► heading-aware chunking ─► embeddings ─► FAISS index
                                                               │
question ─► query embedding ─► top-k similarity search ────────┘
                                   │
                    score < threshold? ─► "I couldn't find this in the study notes."
                                   │
                    grounded prompt + numbered passages ─► LLM ─► cited answer + sources
```

## What it does

- **Ingestion:** loads Markdown, text or PDF notes and splits them into ~120-word overlapping chunks. Each chunk is prefixed with its chapter and section so the embedding keeps its context.
- **Embeddings and vector search:** Gemini `gemini-embedding-001` (768-dim, separate document and query task types) stored in a FAISS inner-product index. Vectors are L2-normalised, so the score is cosine similarity.
- **Grounded generation:** the LLM may only use the retrieved passages and must cite each fact as `[1]`, `[2]`. The API returns only the passages the answer actually cites.
- **Guardrail against hallucination:** if no chunk passes the similarity threshold, the app refuses without calling the LLM, which also saves cost.
- **Subject filter:** restrict search to Biology, Physics or Chemistry.
- **Offline mode:** with no API key it falls back to TF-IDF retrieval and an extractive answerer, so it runs and tests anywhere.
- **Evaluation:** a labelled question set measures retrieval hit@k, MRR, answer accuracy and refusal accuracy.

## Results

On the 24-question evaluation set (20 in scope, 4 out of scope), in offline mode:

| Metric | Value |
|---|---|
| Retrieval hit@4 | 100% |
| Retrieval MRR | 1.00 |
| Answer accuracy (in scope) | 95% |
| Refusal accuracy | 100% |

Run `python -m eval.run_eval` with a Gemini key set to measure the full LLM mode; the report is written to `eval/report.md`.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY (free at aistudio.google.com) — optional
python -m app.ingest          # build the index from data/notes
uvicorn app.main:app --reload # open http://localhost:8000
```

Ask from the command line:

```bash
curl -X POST localhost:8000/api/ask -H "Content-Type: application/json" \
     -d '{"question": "Why is O2 paramagnetic?", "subject": "Chemistry"}'
```

Tests (offline, no key needed): `pytest -q`

## Use your own notes

Drop `.md`, `.txt` or `.pdf` files into `data/notes/` and run `python -m app.ingest` again. Name files with a subject prefix (`biology_…`, `physics_…`, `chemistry_…`) so the subject filter works.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ask` | `{question, subject?}` → `{answer, sources[], in_scope, latency_ms, provider}` |
| GET | `/api/stats` | Chunks, files, subjects, embedder and LLM in use |
| GET | `/api/health` | Readiness check |
| GET | `/docs` | Interactive Swagger docs |

## Deploy

The included `Dockerfile` builds the index on start-up. On Render, connect the repo (it reads `render.yaml`) and set `GEMINI_API_KEY`.

## Project layout

```
app/
  chunking.py     load notes, heading-aware overlapping chunks
  embeddings.py   Gemini and local TF-IDF embedders
  vectorstore.py  FAISS index + metadata, save/load
  ingest.py       CLI to build the index
  rag.py          retrieve → threshold → generate → cited sources
  llm.py          Gemini / Claude / extractive answerers, grounded prompt
  main.py         FastAPI app
web/index.html    chat UI with citations and source excerpts
eval/             labelled questions + evaluation script
tests/            offline pytest suite
```

## Next steps

- Hybrid search (BM25 + vectors) and a re-ranker for harder questions
- Conversation memory for follow-up questions
- Answer streaming and per-question feedback to grow the evaluation set
