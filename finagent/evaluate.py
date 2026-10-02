"""LLM-as-judge grading and the metrics reported in results/RESULTS.md."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from finagent import prompts
from finagent.llm import structured


class Judgement(BaseModel):
    label: Literal["correct", "incorrect", "refusal"]
    reason: str


def judge(llm, question: str, gold: str, justification: str | None, pred: str) -> Judgement:
    # A runaway answer (small local models sometimes ramble) must not blow the judge's token limit.
    if len(pred) > 4000:
        pred = pred[:4000] + " [...answer truncated]"
    return structured(llm, Judgement).invoke(prompts.JUDGE.format(
        question=question, gold=gold, justification=justification or "-", pred=pred))


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    rows = [r for r in rows if "label" in r]
    n = len(rows)
    if not n:
        return {"n": 0}
    pct = lambda k: 100 * k / n  # noqa: E731
    correct = sum(r["label"] == "correct" for r in rows)
    refusal = sum(r["label"] == "refusal" for r in rows)
    incorrect = n - correct - refusal
    return {
        "n": n,
        "correct": pct(correct),
        "incorrect": pct(incorrect),          # wrong answer given confidently ("hallucination" in FinanceBench)
        "refusal": pct(refusal),
        # Did the answer cite (or the agent at least retrieve) a page FinanceBench lists as evidence?
        "cited_evidence_page": pct(sum(bool(set(r["pages"]) & set(r["gold_pages"])) for r in rows)),
        "retrieved_evidence_page": pct(sum(bool(set(r["retrieved_pages"]) & set(r["gold_pages"])) for r in rows)),
        "llm_calls": sum(r.get("llm_calls", 0) for r in rows) / n,
        "tokens": sum(r.get("tokens", 0) for r in rows) / n,
        "seconds": sum(r.get("seconds", 0) for r in rows) / n,
        "used_calculator": pct(sum(bool(r.get("calcs")) for r in rows)),
    }


def by_group(rows: list[dict], key: str) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key) or "none"].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def calculator_fixes(with_calc: list[dict], without_calc: list[dict]) -> dict:
    """Questions the agent got right with the calculator and wrong without it (and the reverse)."""
    a = {r["id"]: r for r in with_calc if "label" in r}
    b = {r["id"]: r for r in without_calc if "label" in r}
    common = a.keys() & b.keys()
    fixed = [i for i in common if a[i]["label"] == "correct" and b[i]["label"] != "correct" and a[i].get("calcs")]
    broke = [i for i in common if b[i]["label"] == "correct" and a[i]["label"] != "correct"]
    return {"compared": len(common), "fixed": sorted(fixed), "broke": sorted(broke)}
