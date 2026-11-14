"""Evaluate retrieval and answers against a labelled question set.

Metrics:
  - Hit@k: the expected notes file appears in the top-k retrieved chunks.
  - MRR: mean reciprocal rank of the first chunk from the expected file.
  - Answer accuracy: the answer contains the expected keyword(s).
  - Refusal accuracy: out-of-scope questions get "I couldn't find this", in-scope ones don't.

Usage:  python -m eval.run_eval  [--k 4]
Writes eval/report.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from app.config import settings
from app.rag import DoubtSolver

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()

    solver = DoubtSolver()
    rows = [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]

    hits, rr, correct, refusals_ok, latencies, lines = [], [], [], [], [], []
    for row in rows:
        retrieved = solver.retrieve(row["question"], k=args.k)
        answer = solver.ask(row["question"])
        latencies.append(answer.latency_ms)
        expected = row["source"]
        if expected:
            sources = [c["source"] for c, _ in retrieved]
            rank = sources.index(expected) + 1 if expected in sources else 0
            hits.append(rank > 0)
            rr.append(1 / rank if rank else 0.0)
            ok = answer.in_scope and all(k.lower() in answer.answer.lower() for k in row["keywords"])
            correct.append(ok)
            refusals_ok.append(answer.in_scope)
            status = "PASS" if ok else "FAIL"
        else:
            refusals_ok.append(not answer.in_scope)
            status = "PASS" if not answer.in_scope else "FAIL"
        lines.append(f"| {status} | {row['question']} | {answer.answer[:110].replace('|', '/')} |")

    pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%"  # noqa: E731
    summary = {
        f"Retrieval hit@{args.k}": pct(hits),
        "Retrieval MRR": f"{statistics.mean(rr):.2f}",
        "Answer accuracy (in-scope)": pct(correct),
        "Refusal accuracy (all questions)": pct(refusals_ok),
        "Median latency": f"{int(statistics.median(latencies))} ms",
    }
    report = [
        "# Evaluation report",
        "",
        f"Embedder: **{solver.store.embedder_name}** · LLM: **{settings.llm_provider}** · "
        f"{len(rows)} questions ({len(hits)} in scope, {len(rows) - len(hits)} out of scope)",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "| Result | Question | Answer |",
        "|---|---|---|",
        *lines,
    ]
    (HERE / "report.md").write_text("\n".join(report) + "\n")
    for k, v in summary.items():
        print(f"{k:34} {v}")
    print(f"Full report: {HERE / 'report.md'}")


if __name__ == "__main__":
    main()
