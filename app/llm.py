"""Answer generation: Gemini, Claude, or an extractive fallback when no API key is set."""
from __future__ import annotations

import os
import re

import httpx

SYSTEM_PROMPT = (
    "You are a NEET doubt solver for students. Answer ONLY from the numbered context passages. "
    "Cite every fact with its passage number in square brackets, like [1] or [2][3]. "
    "Keep the answer short and exam-focused: lead with the direct answer, then at most 3 supporting points. "
    "If the context does not contain the answer, reply exactly: "
    "\"I couldn't find this in the study notes.\" Never use outside knowledge."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({p['chapter']} — {p['section']})\n{p['text']}" for i, p in enumerate(passages, start=1)
    )
    return f"Context passages:\n{context}\n\nStudent question: {question}\n\nAnswer:"


def _gemini(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types

    # System instructions are only supported on the v1beta API, so pin it explicitly.
    client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.1),
    )
    return (response.text or "").strip()


def _claude(prompt: str, model: str) -> str:
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 500,
            "temperature": 0.1,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return "".join(block.get("text", "") for block in response.json()["content"]).strip()


_STOP = set("what which who whom whose why how when where is are was were the a an of in on to for by and or do does did with from as at that this these those it its be".split())


def _stem(word: str) -> str:
    """Very light stemming so 'absorbed', 'absorption' and 'absorb' match."""
    for suffix in ("ation", "ption", "tion", "ing", "ed", "es", "s", "ty"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)][:6]
    return word[:6]


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def extractive_answer(question: str, passages: list[dict]) -> str:
    """No-LLM fallback: return the sentences that best match the question, with citations.

    Terms containing digits (SF6, F2, 104.5) are rare and specific, so they count double.
    """
    q_terms = _terms(question)
    scored = []
    for p_idx, passage in enumerate(passages, start=1):
        body = passage["text"].split(": ", 1)[-1]
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            overlap = q_terms & _terms(sentence)
            score = sum(2 if any(ch.isdigit() for ch in t) else 1 for t in overlap)
            if score:
                # Prefer higher scores, then earlier (more relevant) passages, then shorter sentences.
                scored.append((score, -p_idx, -len(sentence), sentence.strip(), p_idx))
    if not scored:
        return "I couldn't find this in the study notes."
    scored.sort(reverse=True)
    return " ".join(f"{s} [{i}]" for *_, s, i in scored[:2])


def generate(question: str, passages: list[dict], provider: str, gemini_model: str, claude_model: str) -> str:
    if provider == "gemini":
        return _gemini(build_prompt(question, passages), gemini_model)
    if provider == "anthropic":
        return _claude(build_prompt(question, passages), claude_model)
    return extractive_answer(question, passages)
