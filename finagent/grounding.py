"""Deterministic check that the numbers in an answer come from the cited text or from a calculation.

This runs before any LLM-based verification: if every figure in the answer can be traced to the
evidence (allowing for rounding and million/billion rescaling), no extra LLM call is needed.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"\(?-?\$?\d[\d,]*\.?\d*\)?")


def extract_numbers(text: str, skip_years: bool = False) -> list[float]:
    nums = []
    for m in _NUM.finditer(text):
        raw = m.group().rstrip(".")
        if skip_years and re.fullmatch(r"(19|20)\d\d", raw):   # bare 4-digit years, not "$2,000"
            continue
        neg = raw.startswith("(") and raw.endswith(")") or "-" in raw
        digits = re.sub(r"[^\d.]", "", raw).rstrip(".")
        if not digits or digits.count(".") > 1:
            continue
        val = float(digits)
        nums.append(-val if neg else val)
    return nums


def _matches(x: float, y: float) -> bool:
    if x == 0 or y == 0:
        return x == y
    for scale in (1, 1e3, 1e-3, 1e6, 1e-6, 100, 0.01):   # unit changes and % <-> fraction
        z = y * scale
        # Answers round: 1,577 -> 1.58 billion, 0.1486 -> 14.9%. Allow rounding at 2 significant figures.
        if abs(abs(x) - abs(z)) <= max(0.006 * abs(z), 0.051 if abs(z) < 100 else 0):
            return True
    return False


def ungrounded_numbers(answer: str, evidence: str, calc_results: list[str], question: str = "") -> list[float]:
    """Numbers in `answer` that appear neither in the evidence, the question, nor any calculator output."""
    pool = extract_numbers(evidence) + extract_numbers(question)
    for r in calc_results:
        pool += extract_numbers(r)
    missing = []
    for x in extract_numbers(answer, skip_years=True):
        if abs(x) < 10 and x == int(x):   # small counts like "2 segments"
            continue
        if not any(_matches(x, y) for y in pool):
            missing.append(x)
    return missing
