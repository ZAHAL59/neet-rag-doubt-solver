"""Offline tests: they use the local TF-IDF embedder and the extractive answerer, so no API key is needed."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chunking import chunk_document, load_chunks
from app.config import Settings
from app.ingest import build_index
from app.rag import REFUSAL, DoubtSolver

NOTES = Path(__file__).resolve().parent.parent / "data" / "notes"


@pytest.fixture(scope="session")
def solver(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("index")
    build_index(NOTES, index_dir, "local")
    cfg = Settings(index_dir=index_dir, embedder="local", llm_provider="none")
    return DoubtSolver(index_dir, cfg)


def test_chunks_keep_heading_context():
    chunks = chunk_document(NOTES / "biology_photosynthesis.md", chunk_words=60, overlap=15)
    assert chunks, "expected at least one chunk"
    assert all(c.subject == "Biology" for c in chunks)
    assert any(c.section == "C4 pathway" for c in chunks)
    assert all(c.text.startswith("Photosynthesis in Higher Plants — ") for c in chunks)


def test_long_sections_are_split_with_overlap():
    chunks = chunk_document(NOTES / "biology_genetics.md", chunk_words=40, overlap=10)
    mendel = [c for c in chunks if c.section == "Mendel's laws"]
    assert len(mendel) > 1
    first_tail = mendel[0].text.split()[-10:]
    assert " ".join(first_tail) in mendel[1].text


def test_all_notes_load():
    sources = {c.source for c in load_chunks(NOTES)}
    assert len(sources) == 6


@pytest.mark.parametrize(
    "question,expected_source",
    [
        ("What is the first stable product of the C4 pathway?", "biology_photosynthesis.md"),
        ("Which enzyme activates trypsinogen?", "biology_digestion.md"),
        ("Why is O2 paramagnetic?", "chemistry_chemical_bonding.md"),
        ("Do action and reaction forces cancel each other?", "physics_laws_of_motion.md"),
    ],
)
def test_retrieval_finds_the_right_notes(solver, question, expected_source):
    top = solver.retrieve(question, k=3)
    assert top[0][0]["source"] == expected_source


def test_answer_is_cited_and_grounded(solver):
    result = solver.ask("Which enzyme activates trypsinogen?")
    assert result.in_scope
    assert "enterokinase" in result.answer.lower()
    assert "[1]" in result.answer
    assert result.sources and result.sources[0]["source"] == "biology_digestion.md"


def test_out_of_scope_question_is_refused(solver):
    result = solver.ask("What is the capital of Australia?")
    assert not result.in_scope
    assert result.answer == REFUSAL
    assert result.sources == []


def test_subject_filter(solver):
    hits = solver.retrieve("bond", k=3, subject="Chemistry")
    assert hits and all(c["subject"] == "Chemistry" for c, _ in hits)


def test_api_endpoints(solver, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "get_solver", lambda: solver)
    client = TestClient(main.app)
    assert client.get("/").status_code == 200
    response = client.post("/api/ask", json={"question": "What is the bond angle in water?"})
    assert response.status_code == 200
    body = response.json()
    assert body["in_scope"] and "104.5" in body["answer"]
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422
