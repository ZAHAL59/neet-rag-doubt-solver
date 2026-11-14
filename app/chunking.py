"""Load notes (Markdown, text or PDF) and split them into overlapping, heading-aware chunks."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

SUBJECTS = {"biology": "Biology", "physics": "Physics", "chemistry": "Chemistry", "botany": "Biology", "zoology": "Biology"}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    subject: str
    chapter: str
    section: str

    def to_dict(self) -> dict:
        return asdict(self)


def _subject_from_name(name: str) -> str:
    prefix = name.lower().split("_")[0]
    return SUBJECTS.get(prefix, "General")


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _sections(text: str, fallback_title: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (chapter title, [(section heading, body), ...]) using Markdown headings."""
    chapter = fallback_title
    sections: list[tuple[str, str]] = []
    current_heading, current_lines = "Introduction", []
    for line in text.splitlines():
        if line.startswith("# "):
            chapter = line[2:].strip()
        elif line.startswith("## "):
            if "".join(current_lines).strip():
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading, current_lines = line[3:].strip(), []
        else:
            current_lines.append(line)
    if "".join(current_lines).strip():
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return chapter, sections


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def chunk_document(path: Path, chunk_words: int = 120, overlap: int = 30) -> list[Chunk]:
    text = _read(path)
    title = path.stem.replace("_", " ").title()
    chapter, sections = _sections(text, title)
    subject = _subject_from_name(path.stem)
    chunks: list[Chunk] = []
    for s_idx, (heading, body) in enumerate(sections):
        words = re.sub(r"\s+", " ", body).split(" ")
        for w_idx, window in enumerate(_windows(words, chunk_words, overlap)):
            # Prefix the heading so each chunk carries its own context into the embedding.
            chunk_text = f"{chapter} — {heading}: {' '.join(window)}"
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}#{s_idx}.{w_idx}",
                    text=chunk_text,
                    source=path.name,
                    subject=subject,
                    chapter=chapter,
                    section=heading,
                )
            )
    return chunks


def load_chunks(notes_dir: Path, chunk_words: int = 120, overlap: int = 30) -> list[Chunk]:
    files = sorted(p for p in notes_dir.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"})
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_document(path, chunk_words, overlap))
    return chunks
