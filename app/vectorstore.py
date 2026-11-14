"""FAISS vector index plus chunk metadata, saved to and loaded from disk."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[dict], embedder_name: str) -> None:
        self.index = index
        self.chunks = chunks
        self.embedder_name = embedder_name

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list[Chunk], embedder_name: str) -> "VectorStore":
        # Vectors are L2-normalised, so inner product equals cosine similarity.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, [c.to_dict() for c in chunks], embedder_name)

    def search(self, query_vector: np.ndarray, k: int, subject: str | None = None) -> list[tuple[dict, float]]:
        # Over-fetch when filtering by subject so enough results survive the filter.
        fetch = min(len(self.chunks), k * 4 if subject else k)
        scores, ids = self.index.search(query_vector, fetch)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if subject and chunk["subject"].lower() != subject.lower():
                continue
            results.append((chunk, float(score)))
            if len(results) == k:
                break
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        meta = {"embedder": self.embedder_name, "dim": self.index.d, "chunks": self.chunks}
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        index = faiss.read_index(str(directory / "index.faiss"))
        meta = json.loads((directory / "chunks.json").read_text())
        return cls(index, meta["chunks"], meta["embedder"])
