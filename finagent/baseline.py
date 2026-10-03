"""Plain RAG for comparison: one search with the raw question, one LLM call, no tools, no checks.

It uses the same retriever and the same main model as the agent, so any difference in the
evaluation comes from the agent's loop (planning, grading, rewriting, calculator, verification).
"""

from __future__ import annotations

import re

from finagent import config, prompts
from finagent.graph import _is_not_found, format_chunks


def run_baseline(llm, question: str, doc_name: str, search_fn=None, k: int = 8, callbacks=None) -> dict:
    if search_fn is None:
        from finagent.retriever import search as search_fn
    chunks = search_fn(question, doc_name, k=k)
    msg = llm.invoke(prompts.BASELINE.format(
        doc_name=doc_name, not_found=config.NOT_FOUND, evidence=format_chunks(chunks), question=question),
        config={"callbacks": callbacks} if callbacks else None)  # None keeps callbacks inherited from a graph run
    text = msg.text if hasattr(msg, "text") else str(msg.content)

    pages = []
    m = re.search(r"pages?\s*:\s*(.*)$", text, flags=re.I | re.M)
    if m:
        pages = [int(p) - 1 for p in re.findall(r"\d+", m.group(1))]
        text = text[: m.start()].strip()
    retrieved = sorted({c.metadata["page"] for c in chunks})
    not_found = _is_not_found(text)
    return {
        "answer": config.NOT_FOUND if not_found else text,
        "status": "not_found" if not_found else "answered",
        "pages": [] if not_found else sorted({p for p in pages if p in retrieved}) or retrieved,
        "retrieved_pages": retrieved,
        "calcs": [],
        "trace": [],
    }
