"""Admin document management: add an uploaded file to the knowledge base, list files, delete them."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .chunking import SUPPORTED, chunk_document
from .config import settings
from .embeddings import GeminiEmbedder
from .ingest import build_index
from .vectorstore import VectorStore

MAX_BYTES = 15 * 1024 * 1024  # 15 MB per upload


class UploadError(ValueError):
    pass


# ---------- Optional: keep uploads permanently in the GitHub repo ----------
# Free hosting (e.g. Render free) wipes the disk on restart. If GITHUB_TOKEN and GITHUB_REPO are set,
# every upload and delete is also committed to data/notes/ in the repo, so the index is rebuilt with it.


def _github_target() -> tuple[str, str, str] | None:
    import os

    token, repo = os.getenv("GITHUB_TOKEN"), os.getenv("GITHUB_REPO")
    return (token, repo, os.getenv("GITHUB_BRANCH", "main")) if token and repo else None


def _github_sync(name: str, data: bytes | None) -> str:
    """Create/update (data given) or delete (data None) data/notes/<name> in the repo. Returns a status."""
    import base64

    import httpx

    target = _github_target()
    if not target:
        return "not configured"
    token, repo, branch = target
    url = f"https://api.github.com/repos/{repo}/contents/data/notes/{name}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        existing = httpx.get(url, headers=headers, params={"ref": branch}, timeout=20)
        sha = existing.json().get("sha") if existing.status_code == 200 else None
        if data is None:
            if not sha:
                return "not in repo"
            r = httpx.request("DELETE", url, headers=headers, timeout=30,
                              json={"message": f"Remove notes: {name}", "sha": sha, "branch": branch})
        else:
            body = {"message": f"{'Update' if sha else 'Add'} notes: {name}", "branch": branch,
                    "content": base64.b64encode(data).decode()}
            if sha:
                body["sha"] = sha
            r = httpx.put(url, headers=headers, json=body, timeout=30)
        return "saved to GitHub" if r.status_code in (200, 201) else f"GitHub error {r.status_code}"
    except httpx.HTTPError as exc:
        return f"GitHub unreachable ({type(exc).__name__})"


def safe_filename(original: str, subject: str) -> str:
    """Make a clean file name and prefix the subject so the subject filter works (e.g. biology_respiration.pdf)."""
    stem, suffix = Path(original).stem, Path(original).suffix.lower()
    if suffix not in SUPPORTED:
        raise UploadError(f"Unsupported file type {suffix or '(none)'}. Use PDF, DOCX, MD or TXT.")
    stem = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:60] or "document"
    prefix = subject.lower()
    if prefix not in {"biology", "physics", "chemistry"}:
        raise UploadError("Subject must be Biology, Physics or Chemistry.")
    if not stem.startswith(prefix + "_"):
        stem = f"{prefix}_{stem}"
    return stem + suffix


def add_document(data: bytes, original_name: str, subject: str, notes_dir: Path | None = None, index_dir: Path | None = None) -> dict:
    notes_dir = notes_dir or settings.notes_dir
    index_dir = index_dir or settings.index_dir
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadError("The file is larger than 15 MB.")

    name = safe_filename(original_name, subject)
    path = notes_dir / name
    replaced = path.exists()
    notes_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

    chunks = chunk_document(path, settings.chunk_words, settings.chunk_overlap)
    if not chunks or sum(len(c.text) for c in chunks) < 200:
        path.unlink(missing_ok=True)
        raise UploadError("Couldn't extract enough text. Scanned PDFs (images of pages) need OCR first.")

    store = VectorStore.load(index_dir)
    if store.embedder_name == "gemini":
        # Gemini embeddings are fixed, so new chunks can be appended without touching the rest.
        if replaced:
            store.remove_source(name)
        embedder = GeminiEmbedder(settings.gemini_embed_model, settings.embed_dim)
        store.add(embedder.embed_documents([c.text for c in chunks]), chunks)
        store.save(index_dir)
    else:
        # TF-IDF's vocabulary depends on the whole corpus, so the local mode rebuilds everything.
        store = build_index(notes_dir, index_dir, "local")

    return {
        "file": name,
        "chapter": chunks[0].chapter,
        "sections": len({c.section for c in chunks}),
        "chunks": len(chunks),
        "replaced": replaced,
        "total_chunks": len(store.chunks),
        "storage": _github_sync(name, data),
    }


def delete_document(name: str, notes_dir: Path | None = None, index_dir: Path | None = None) -> dict:
    notes_dir = notes_dir or settings.notes_dir
    index_dir = index_dir or settings.index_dir
    path = notes_dir / Path(name).name  # never allow paths outside the notes folder
    store = VectorStore.load(index_dir)
    if not path.exists() and not any(c["source"] == path.name for c in store.chunks):
        raise UploadError(f"No document named {name}.")
    path.unlink(missing_ok=True)
    if store.embedder_name == "gemini":
        removed = store.remove_source(path.name)
        store.save(index_dir)
    else:
        before = len(store.chunks)
        store = build_index(notes_dir, index_dir, "local")
        removed = before - len(store.chunks)
    return {
        "file": path.name,
        "removed_chunks": removed,
        "total_chunks": len(store.chunks),
        "storage": _github_sync(path.name, None),
    }


def list_documents(index_dir: Path | None = None) -> list[dict]:
    store = VectorStore.load(index_dir or settings.index_dir)
    docs: dict[str, dict] = defaultdict(lambda: {"chunks": 0, "sections": set()})
    for c in store.chunks:
        d = docs[c["source"]]
        d.update(file=c["source"], subject=c["subject"], chapter=c["chapter"])
        d["chunks"] += 1
        d["sections"].add(c["section"])
    return sorted(
        ({**d, "sections": len(d["sections"])} for d in docs.values()),
        key=lambda d: (d["subject"], d["chapter"]),
    )
