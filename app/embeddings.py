"""Embedding back ends: Google Gemini embeddings, or a local TF-IDF model that needs no API key."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class LocalEmbedder:
    """TF-IDF vectors (word 1–2 grams). Used for offline development and tests."""

    name = "local"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            self.fit(texts)
        return _normalize(self.vectorizer.transform(texts).toarray())

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self.vectorizer.transform([text]).toarray())

    def save(self, directory: Path) -> None:
        with open(directory / "tfidf.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, directory: Path) -> "LocalEmbedder":
        obj = cls()
        with open(directory / "tfidf.pkl", "rb") as f:
            obj.vectorizer = pickle.load(f)
        obj.fitted = True
        return obj


class GeminiEmbedder:
    """Google Gemini embeddings, with separate task types for documents and queries."""

    name = "gemini"

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768) -> None:
        from google import genai

        self.client = genai.Client()  # reads GEMINI_API_KEY from the environment
        self.model = model
        self.dim = dim

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        vectors = []
        for start in range(0, len(texts), 64):  # batch to stay within request limits
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts[start : start + 64],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
            )
            vectors.extend(e.values for e in result.embeddings)
        return _normalize(np.array(vectors, dtype="float32"))

    def fit(self, texts: list[str]) -> None:  # nothing to train
        pass

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")

    def save(self, directory: Path) -> None:
        pass

    @classmethod
    def load(cls, directory: Path, model: str = "gemini-embedding-001", dim: int = 768) -> "GeminiEmbedder":
        return cls(model, dim)


def make_embedder(kind: str, model: str = "gemini-embedding-001", dim: int = 768):
    if kind == "gemini":
        return GeminiEmbedder(model, dim)
    if kind == "local":
        return LocalEmbedder()
    raise ValueError(f"Unknown embedder: {kind!r} (use 'gemini' or 'local')")
