"""Keeping a public demo alive on free-tier APIs.

- Saved runs: the evaluation already answered 145 FinanceBench questions; replaying those costs nothing.
- Answer cache: a question asked once about a report is answered from disk the next time.
- Fallback chain: when one model hits its free-tier limit, the next model (with its own quota) takes over.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from finagent import config


@dataclass(frozen=True)
class ModelConfig:
    label: str
    main: str
    fast: str
    key_env: str | None     # environment variable holding the provider's key


# Each step uses models with their own free-tier quota, so one running out does not stop the next.
DEMO_CHAIN = [
    ModelConfig("gpt-oss-120b (Groq)", "groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b", "GROQ_API_KEY"),
    ModelConfig("gpt-oss-20b (Groq)", "groq:openai/gpt-oss-20b", "groq:qwen/qwen3.8-27b", "GROQ_API_KEY"),
    ModelConfig("qwen3.8-27b (Groq)", "groq:qwen/qwen3.8-27b", "groq:qwen/qwen3.8-27b", "GROQ_API_KEY"),
    ModelConfig("Gemini 3.5 Flash", "google_genai:gemini-3.5-flash", "google_genai:gemini-3.5-flash-lite",
                "GOOGLE_API_KEY"),
]

_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "resource_exhausted", "quota", "tokens per day",
                  "requests per day", "request too large", "413")


def is_rate_limit(exc: BaseException) -> bool:
    """True for provider errors that mean 'this model's free quota is used up (for now)'."""
    text = f"{type(exc).__name__} {exc}".lower()
    return "ratelimit" in type(exc).__name__.lower() or any(m in text for m in _LIMIT_MARKERS)


class AllModelsBusy(Exception):
    """Every model in the chain is at its free-tier limit."""


def run_with_fallback(chain, run_one, on_switch=None):
    """Call run_one(cfg) for each config until one succeeds without hitting a rate limit.

    Returns (result, cfg). Errors that are not rate limits are raised immediately: falling back would
    only hide a real bug. on_switch(failed_cfg, next_cfg) lets the UI say what happened.
    """
    chain = list(chain)
    for i, cfg in enumerate(chain):
        try:
            return run_one(cfg), cfg
        except Exception as e:
            if not is_rate_limit(e):
                raise
            if on_switch and i + 1 < len(chain):
                on_switch(cfg, chain[i + 1])
    raise AllModelsBusy("every model in the demo chain is at its free-tier limit")


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip("?. ")


class AnswerCache:
    """Answers already given, keyed by report and question, shared by every visitor."""

    def __init__(self, path: Path = config.DATA_DIR / "answer_cache.sqlite"):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT, created REAL)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def key(doc_name: str, question: str) -> str:
        return hashlib.sha256(f"{doc_name}\n{normalize(question)}".encode()).hexdigest()

    def get(self, doc_name: str, question: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT value FROM answers WHERE key = ?", (self.key(doc_name, question),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, doc_name: str, question: str, result: dict) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)",
                       (self.key(doc_name, question), json.dumps(result, ensure_ascii=False), time.time()))


def ensure_index() -> bool:
    """On a fresh host, download the prebuilt demo index from the Hugging Face Hub.

    Set FINAGENT_INDEX_REPO to a dataset repo (e.g. "user/finreport-demo-index") holding the contents of
    data/demo_chroma. Returns True when an index is available locally.
    """
    import os

    if config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir()):
        return True
    repo = os.getenv("FINAGENT_INDEX_REPO")
    if not repo:
        return False
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(config.CHROMA_DIR))
    return True


def load_saved_runs(results_dir: Path = config.RESULTS_DIR, tag: str = "") -> dict[tuple[str, str], dict]:
    """Agent answers from the evaluation, keyed by (report, normalized question)."""
    path = results_dir / f"agent{tag}.jsonl"
    if not path.exists():
        return {}
    saved = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or "label" not in r:
            continue
        saved[(r["doc_name"], normalize(r["question"]))] = r
    return saved
