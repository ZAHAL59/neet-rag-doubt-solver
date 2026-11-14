"""FastAPI server: a JSON API plus a small chat page.

Run:  uvicorn app.main:app --reload
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .rag import DoubtSolver

WEB = Path(__file__).resolve().parent.parent / "web" / "index.html"

app = FastAPI(title="NEET Doubt Solver", version="1.0.0", description="RAG over NEET study notes, with cited answers.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    subject: str | None = Field(None, description="Optional filter: Biology, Physics or Chemistry")


@lru_cache(maxsize=1)
def get_solver() -> DoubtSolver:
    if not (settings.index_dir / "index.faiss").exists():
        raise HTTPException(503, "Index not built yet. Run: python -m app.ingest")
    return DoubtSolver()


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception) -> JSONResponse:
    # Always return JSON so the chat page can show a readable message.
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"[:300]})


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(WEB)


@app.get("/api/health")
def health() -> dict:
    ready = (settings.index_dir / "index.faiss").exists()
    return {"status": "ok" if ready else "index_missing", "llm_provider": settings.llm_provider}


@app.get("/api/stats")
def stats() -> dict:
    solver = get_solver()
    chunks = solver.store.chunks
    return {
        "chunks": len(chunks),
        "files": len({c["source"] for c in chunks}),
        "subjects": dict(Counter(c["subject"] for c in chunks)),
        "chapters": sorted({c["chapter"] for c in chunks}),
        "embedder": solver.store.embedder_name,
        "llm_provider": settings.llm_provider,
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    return get_solver().ask(req.question, req.subject).to_dict()
