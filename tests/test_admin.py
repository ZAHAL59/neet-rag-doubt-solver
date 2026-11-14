"""Admin upload tests: they run offline with the local embedder and a temporary copy of the notes."""
import io
import shutil
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import library, main
from app.config import Settings
from app.ingest import build_index
from app.rag import DoubtSolver

NOTES = Path(__file__).resolve().parent.parent / "data" / "notes"

NEW_NOTE = b"""# Breathing and Exchange of Gases

## Respiratory pigments
Haemoglobin is the respiratory pigment in human red blood cells. About 97 percent of oxygen is transported by
haemoglobin as oxyhaemoglobin, and only about 3 percent is carried dissolved in plasma.

## Carbon dioxide transport
About 70 percent of carbon dioxide is carried as bicarbonate ions in the plasma. The enzyme carbonic anhydrase,
found in red blood cells, speeds up the reaction between carbon dioxide and water.
"""


def make_docx(paragraphs: list[tuple[str, str]]) -> bytes:
    """Build a minimal .docx with (style, text) paragraphs."""
    body = "".join(
        f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>' if style
        else f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
        for style, text in paragraphs
    )
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    notes, index = tmp_path / "notes", tmp_path / "index"
    shutil.copytree(NOTES, notes)
    build_index(notes, index, "local")
    cfg = Settings(notes_dir=notes, index_dir=index, embedder="local", llm_provider="none")
    monkeypatch.setattr(library, "settings", cfg)
    monkeypatch.setattr(main, "settings", cfg)
    main.get_solver.cache_clear()
    monkeypatch.setattr(main, "DoubtSolver", lambda: DoubtSolver(index, cfg))
    monkeypatch.setenv("ADMIN_TOKEN", "test-secret")
    yield TestClient(main.app)
    main.get_solver.cache_clear()


AUTH = {"X-Admin-Token": "test-secret"}


def test_upload_requires_the_admin_password(env):
    files = {"file": ("respiration.md", NEW_NOTE, "text/markdown")}
    assert env.post("/api/admin/documents", files=files, data={"subject": "Biology"}).status_code == 401
    bad = {"X-Admin-Token": "wrong"}
    assert env.get("/api/admin/documents", headers=bad).status_code == 401


def test_uploaded_note_is_answerable_immediately(env):
    before = env.post("/api/ask", json={"question": "Which enzyme speeds up carbon dioxide transport?"}).json()
    assert "carbonic anhydrase" not in before["answer"].lower()

    files = {"file": ("Respiration Notes.md", NEW_NOTE, "text/markdown")}
    r = env.post("/api/admin/documents", files=files, data={"subject": "Biology"}, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["file"] == "biology_respiration_notes.md"
    assert body["chapter"] == "Breathing and Exchange of Gases" and body["chunks"] >= 2

    after = env.post("/api/ask", json={"question": "Which enzyme speeds up carbon dioxide transport?"}).json()
    assert after["in_scope"] and "carbonic anhydrase" in after["answer"].lower()
    assert after["sources"][0]["source"] == "biology_respiration_notes.md"


def test_docx_headings_become_sections(env):
    doc = make_docx([
        ("Heading1", "Excretory Products"),
        ("Heading2", "Nephron"),
        ("", "The nephron is the structural and functional unit of the kidney. Each kidney has about one million nephrons."),
        ("Heading2", "Ultrafiltration"),
        ("", "Ultrafiltration happens in the glomerulus, where blood is filtered under pressure into Bowman's capsule. "
             "About 180 litres of glomerular filtrate are formed every day."),
    ])
    r = env.post("/api/admin/documents", files={"file": ("excretion.docx", doc)}, data={"subject": "Biology"}, headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["chapter"] == "Excretory Products" and r.json()["sections"] == 2
    ans = env.post("/api/ask", json={"question": "Where does ultrafiltration happen?"}).json()
    assert ans["in_scope"] and "glomerulus" in ans["answer"].lower()


def test_list_and_delete(env):
    env.post("/api/admin/documents", files={"file": ("respiration.md", NEW_NOTE)}, data={"subject": "Biology"}, headers=AUTH)
    docs = env.get("/api/admin/documents", headers=AUTH).json()["documents"]
    assert any(d["file"] == "biology_respiration.md" for d in docs)

    r = env.delete("/api/admin/documents/biology_respiration.md", headers=AUTH)
    assert r.status_code == 200 and r.json()["removed_chunks"] >= 2
    docs = env.get("/api/admin/documents", headers=AUTH).json()["documents"]
    assert not any(d["file"] == "biology_respiration.md" for d in docs)


def test_rejects_bad_files(env):
    r = env.post("/api/admin/documents", files={"file": ("virus.exe", b"MZ...")}, data={"subject": "Biology"}, headers=AUTH)
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]
    r = env.post("/api/admin/documents", files={"file": ("tiny.md", b"hi")}, data={"subject": "Biology"}, headers=AUTH)
    assert r.status_code == 400
    r = env.post("/api/admin/documents", files={"file": ("x.md", NEW_NOTE)}, data={"subject": "History"}, headers=AUTH)
    assert r.status_code == 400


def test_admin_disabled_without_token(env, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN")
    assert env.get("/api/admin/documents", headers=AUTH).status_code == 403


def test_vectorstore_remove_rebuilds_index(tmp_path):
    from app.vectorstore import VectorStore

    build_index(NOTES, tmp_path, "local")
    store = VectorStore.load(tmp_path)
    total = store.index.ntotal
    removed = store.remove_source("biology_cell.md")
    assert removed > 0 and store.index.ntotal == total - removed == len(store.chunks)


def test_github_sync_commits_uploads(env, monkeypatch):
    calls = []

    class Resp:
        def __init__(self, code, payload=None):
            self.status_code, self._p = code, payload or {}

        def json(self):
            return self._p

    import httpx

    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPO", "me/repo")
    monkeypatch.setattr(httpx, "get", lambda url, **k: (calls.append(("GET", url)), Resp(404))[1])
    monkeypatch.setattr(httpx, "put", lambda url, **k: (calls.append(("PUT", url, k["json"]["message"])), Resp(201))[1])
    r = env.post("/api/admin/documents", files={"file": ("respiration.md", NEW_NOTE)}, data={"subject": "Biology"}, headers=AUTH)
    assert r.json()["storage"] == "saved to GitHub"
    assert calls[-1] == ("PUT", "https://api.github.com/repos/me/repo/contents/data/notes/biology_respiration.md", "Add notes: biology_respiration.md")
