"""Build the vector index from the notes folder.

Usage:
    python -m app.ingest                # uses EMBEDDER from settings
    python -m app.ingest --embedder local
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .chunking import load_chunks
from .config import settings
from .embeddings import make_embedder
from .vectorstore import VectorStore


def build_index(notes_dir: Path, index_dir: Path, embedder_kind: str) -> VectorStore:
    chunks = load_chunks(notes_dir, settings.chunk_words, settings.chunk_overlap)
    if not chunks:
        raise SystemExit(f"No .md, .txt or .pdf notes found in {notes_dir}")
    embedder = make_embedder(embedder_kind, settings.gemini_embed_model, settings.embed_dim)
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.embed_documents(texts)
    store = VectorStore.build(vectors, chunks, embedder.name)
    store.save(index_dir)
    embedder.save(index_dir)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Index study notes for the NEET doubt solver.")
    parser.add_argument("--notes", type=Path, default=settings.notes_dir)
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--embedder", default=settings.embedder, choices=["gemini", "local"])
    args = parser.parse_args()

    start = time.perf_counter()
    store = build_index(args.notes, args.out, args.embedder)
    files = len({c["source"] for c in store.chunks})
    print(
        f"Indexed {len(store.chunks)} chunks from {files} files "
        f"with the {store.embedder_name} embedder (dim {store.index.d}) "
        f"in {time.perf_counter() - start:.1f}s -> {args.out}"
    )


if __name__ == "__main__":
    main()
