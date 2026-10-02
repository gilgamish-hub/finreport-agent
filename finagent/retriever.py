"""Hybrid retrieval inside one report: dense (bge-small in Chroma) + BM25, fused with reciprocal rank.

BM25 matters here: questions often name an exact line item ("purchases of property, plant and
equipment") that keyword search finds more reliably than embeddings do.
"""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from finagent import config
from finagent.ingest import get_vectorstore

_STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was were
what which with how much many did does do fy""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@lru_cache(maxsize=16)
def _doc_index(doc_name: str) -> tuple[list[Document], BM25Okapi | None]:
    got = get_vectorstore().get(where={"doc_name": doc_name}, include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(got["documents"], got["metadatas"])]
    bm25 = BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None
    return docs, bm25


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def search(query: str, doc_name: str, k: int = config.TOP_K) -> list[Document]:
    docs, bm25 = _doc_index(doc_name)
    if not docs:
        raise ValueError(f"{doc_name} is not indexed. Run scripts/build_index.py first.")
    by_id = {d.metadata["chunk_id"]: d for d in docs}

    dense = get_vectorstore().similarity_search(query, k=k * 3, filter={"doc_name": doc_name})
    dense_ids = [d.metadata["chunk_id"] for d in dense]

    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[: k * 3]
    sparse_ids = [docs[i].metadata["chunk_id"] for i in top if scores[i] > 0]

    return [by_id[cid] for cid in rrf([dense_ids, sparse_ids])[:k] if cid in by_id]


def _strip_header(text: str) -> str:
    return text.split("\n", 1)[1] if text.startswith("[") else text


def _join_overlapping(a: str, b: str, max_overlap: int = config.CHUNK_OVERLAP + 50) -> str:
    for k in range(min(max_overlap, len(a), len(b)), 0, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + "\n" + b


def page_text(doc_name: str, page: int) -> str:
    docs, _ = _doc_index(doc_name)
    parts = sorted((d for d in docs if d.metadata["page"] == page),
                   key=lambda d: int(d.metadata["chunk_id"].rsplit(":", 1)[1]))
    text = ""
    for p in parts:
        body = _strip_header(p.page_content)
        text = _join_overlapping(text, body) if text else body
    return f"[{doc_name} | page {page + 1}]\n{text}"


def expand_to_pages(chunks: list[Document], max_chars: int = config.MAX_EVIDENCE_CHARS) -> list[Document]:
    """Replace chunks by the full pages they come from ("small-to-big" retrieval).

    Chunks are small for precise search, but a table is often split across two chunks; the
    reasoning step gets the whole page so it never sees half a table. Pages are added in the
    order the chunks were kept until `max_chars` is reached; after that a page that does not fit
    is represented by the kept chunk alone.
    """
    out, seen, used = [], set(), 0
    for c in chunks:
        key = (c.metadata["doc_name"], c.metadata["page"])
        if key in seen:
            continue
        full = page_text(*key)
        text = full if used + len(full) <= max_chars else c.page_content
        if used + len(text) > max_chars and out:
            continue
        seen.add(key)
        used += len(text)
        out.append(Document(page_content=text, metadata={**c.metadata, "chunk_id": f"{key[0]}:{key[1]}"}))
    return out


def search_many(queries: list[str], doc_name: str, k: int = config.TOP_K) -> list[Document]:
    """Run several queries and interleave the results, dropping duplicates."""
    results = [search(q, doc_name, k) for q in queries]
    seen, merged = set(), []
    for rank in range(k):
        for res in results:
            if rank < len(res) and res[rank].metadata["chunk_id"] not in seen:
                seen.add(res[rank].metadata["chunk_id"])
                merged.append(res[rank])
    return merged
