"""The RAG pipeline: retrieve relevant chunks, refuse when nothing is relevant, otherwise generate a cited answer."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, settings as default_settings
from .embeddings import GeminiEmbedder, LocalEmbedder
from .llm import generate
from .vectorstore import VectorStore

REFUSAL = "I couldn't find this in the study notes."


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    in_scope: bool = True
    latency_ms: int = 0
    provider: str = "none"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class DoubtSolver:
    def __init__(self, index_dir: Path | None = None, cfg: Settings = default_settings) -> None:
        self.cfg = cfg
        index_dir = index_dir or cfg.index_dir
        self.store = VectorStore.load(index_dir)
        if self.store.embedder_name == "gemini":
            self.embedder = GeminiEmbedder.load(index_dir, cfg.gemini_embed_model, cfg.embed_dim)
        else:
            self.embedder = LocalEmbedder.load(index_dir)
        self.min_score = cfg.min_score_for(self.store.embedder_name)

    def retrieve(self, question: str, k: int | None = None, subject: str | None = None) -> list[tuple[dict, float]]:
        return self.store.search(self.embedder.embed_query(question), k or self.cfg.top_k, subject)

    def ask(self, question: str, subject: str | None = None) -> Answer:
        start = time.perf_counter()
        question = question.strip()
        hits = self.retrieve(question, subject=subject)
        relevant = [(c, s) for c, s in hits if s >= self.min_score]

        if not relevant:
            return Answer(question, REFUSAL, [], False, int((time.perf_counter() - start) * 1000), self.cfg.llm_provider)

        passages = [c for c, _ in relevant]
        text = generate(question, passages, self.cfg.llm_provider, self.cfg.gemini_model, self.cfg.claude_model)
        in_scope = REFUSAL.lower() not in text.lower()

        # Return only the passages the answer actually cites (all of them if it cites none).
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        sources = [
            {
                "ref": i,
                "chapter": c["chapter"],
                "section": c["section"],
                "subject": c["subject"],
                "source": c["source"],
                "score": round(s, 3),
                "excerpt": c["text"].split(": ", 1)[-1][:280],
            }
            for i, (c, s) in enumerate(relevant, start=1)
            if not cited or i in cited
        ]
        return Answer(
            question,
            text,
            sources if in_scope else [],
            in_scope,
            int((time.perf_counter() - start) * 1000),
            self.cfg.llm_provider,
        )
