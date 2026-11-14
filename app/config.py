"""Settings, read from environment variables (or a .env file in the project root)."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    notes_dir: Path = Path(os.getenv("NOTES_DIR", ROOT / "data" / "notes"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))

    # "gemini" uses Google's embedding model; "local" uses TF-IDF and needs no API key.
    embedder: str = os.getenv("EMBEDDER", "gemini" if os.getenv("GEMINI_API_KEY") else "local")
    gemini_embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))

    # "gemini", "anthropic", or "none" (extractive answers, no LLM call).
    llm_provider: str = os.getenv(
        "LLM_PROVIDER",
        "gemini" if os.getenv("GEMINI_API_KEY") else ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "none"),
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")

    chunk_words: int = int(os.getenv("CHUNK_WORDS", "120"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "30"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    # Below this similarity the question is treated as out of scope. Gemini embeddings and
    # TF-IDF have very different score ranges, so each has its own default (see min_score_for).
    min_score_override: str | None = os.getenv("MIN_SCORE")

    def min_score_for(self, embedder_name: str) -> float:
        if self.min_score_override:
            return float(self.min_score_override)
        return 0.55 if embedder_name == "gemini" else 0.12


settings = Settings()
