"""FastAPI server: a JSON API, a student chat page and an admin page for managing documents.

Run:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import hmac
import os
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import library
from .config import settings
from .rag import DoubtSolver

WEB = Path(__file__).resolve().parent.parent / "web"
_write_lock = threading.Lock()  # one upload or delete at a time

app = FastAPI(title="NEET Doubt Solver", version="1.1.0", description="RAG over NEET study notes, with cited answers.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    subject: str | None = Field(None, description="Optional filter: Biology, Physics or Chemistry")


@lru_cache(maxsize=1)
def get_solver() -> DoubtSolver:
    if not (settings.index_dir / "index.faiss").exists():
        raise HTTPException(503, "Index not built yet. Run: python -m app.ingest")
    return DoubtSolver()


def require_admin(token: str | None) -> None:
    expected = os.getenv("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(403, "Admin uploads are disabled. Set the ADMIN_TOKEN environment variable to enable them.")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "Wrong admin password.")


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception) -> JSONResponse:
    # Always return JSON so the pages can show a readable message.
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"[:300]})


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(WEB / "admin.html")


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


# ---------- Admin: manage the knowledge base ----------


@app.get("/api/admin/documents")
def admin_documents(x_admin_token: str | None = Header(None)) -> dict:
    require_admin(x_admin_token)
    return {"documents": library.list_documents()}


@app.post("/api/admin/documents")
async def admin_upload(
    file: UploadFile = File(...),
    subject: str = Form(...),
    x_admin_token: str | None = Header(None),
) -> dict:
    require_admin(x_admin_token)
    data = await file.read()
    try:
        with _write_lock:
            result = library.add_document(data, file.filename or "document", subject)
            get_solver.cache_clear()  # reload the index so new content is searchable right away
    except library.UploadError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


@app.delete("/api/admin/documents/{name}")
def admin_delete(name: str, x_admin_token: str | None = Header(None)) -> dict:
    require_admin(x_admin_token)
    try:
        with _write_lock:
            result = library.delete_document(name)
            get_solver.cache_clear()
    except library.UploadError as exc:
        raise HTTPException(404, str(exc)) from exc
    return result
