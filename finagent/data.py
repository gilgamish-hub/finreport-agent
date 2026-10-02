"""FinanceBench questions and the annual-report PDFs they refer to."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from finagent import config


@dataclass
class Question:
    id: str
    company: str
    doc_name: str
    question: str
    answer: str
    question_type: str            # metrics-generated | domain-relevant | novel-generated
    reasoning: str | None         # e.g. "Numerical reasoning", None for domain-relevant
    evidence_pages: list[int] = field(default_factory=list)   # 0-indexed, as in FinanceBench
    justification: str | None = None


def load_questions() -> list[Question]:
    from datasets import load_dataset

    ds = load_dataset(config.HF_DATASET, split="train")
    out = []
    for r in ds:
        out.append(Question(
            id=r["financebench_id"],
            company=r["company"],
            doc_name=r["doc_name"],
            question=r["question"],
            answer=r["answer"],
            question_type=r["question_type"],
            reasoning=r["question_reasoning"],
            evidence_pages=sorted({e["evidence_page_num"] for e in r["evidence"]}),
            justification=r["justification"],
        ))
    return out


@dataclass
class Report:
    doc_name: str
    company: str
    doc_type: str      # 10k, 10q, 8k, earnings
    period: int
    link: str          # the company's original PDF


def load_reports() -> dict[str, Report]:
    from datasets import load_dataset

    ds = load_dataset(config.HF_DATASET, split="train")
    return {r["doc_name"]: Report(r["doc_name"], r["company"], r["doc_type"], r["doc_period"], r["doc_link"])
            for r in ds}


def pdf_path(doc_name: str) -> Path:
    return config.PDF_DIR / f"{doc_name}.pdf"


def download_pdf(doc_name: str, retries: int = 3) -> Path:
    path = pdf_path(doc_name)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = config.PDF_URL.format(doc_name=doc_name)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError(f"{url} did not return a PDF")
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return path
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return path
