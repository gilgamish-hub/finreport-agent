"""PDF -> page text -> chunks (with page numbers) -> Chroma.

Every chunk keeps the report name and the 0-indexed page it came from, so answers can cite pages
and be checked against FinanceBench's evidence pages.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pymupdf as fitz  # PyMuPDF ("import fitz" prints a deprecation notice to stdout)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finagent import config


def clean_page_text(text: str) -> str:
    """Make PyMuPDF's layout-preserving text compact without losing table columns.

    `get_text(sort=True)` keeps each table row on one line but pads cells with long runs of spaces.
    We glue currency signs to their numbers and turn the padding into " | " column separators.
    """
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\$\s+(?=[\d(])", "$", line)          # "$     5,363" -> "$5,363"
        line = re.sub(r"(?<=\S)[ \t]{3,}(?=\S)", " | ", line)  # column padding -> separator
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def load_pdf_pages(path, doc_name: str, company: str = "") -> list[Document]:
    pages = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf):
            text = clean_page_text(page.get_text(sort=True))
            if len(text) < 30:
                continue
            pages.append(Document(
                page_content=text,
                metadata={"doc_name": doc_name, "company": company, "page": i},
            ))
    return pages


def chunk_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " | ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for n, c in enumerate(chunks):
        m = c.metadata
        m["chunk_id"] = f"{m['doc_name']}:{m['page']}:{n}"
        # A short header gives the embedding (and the LLM) the context a bare table fragment lacks.
        c.page_content = f"[{m['doc_name']} | page {m['page'] + 1}]\n{c.page_content}"
    return chunks


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": config.EMBED_QUERY_PROMPT},
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def indexed_docs() -> set[str]:
    # Paged: one get() over all ~36k chunks exceeds SQLite's variable limit.
    store, docs, offset, page = get_vectorstore(), set(), 0, 5000
    while True:
        metas = store.get(include=["metadatas"], limit=page, offset=offset)["metadatas"]
        docs.update(m["doc_name"] for m in metas)
        if len(metas) < page:
            return docs
        offset += page


def index_pdf(path, doc_name: str, company: str = "", progress=None, batch: int = 256) -> int:
    """Add one report to the vector store. Returns the number of chunks written (0 if already there).

    A report left half-written by an interrupted run is deleted and indexed again.
    progress(done, total) is called after each batch, e.g. to drive a progress bar.
    """
    store = get_vectorstore()
    chunks = chunk_pages(load_pdf_pages(path, doc_name, company))
    ids = [c.metadata["chunk_id"] for c in chunks]
    existing = store.get(where={"doc_name": doc_name}, include=[])["ids"]
    if len(existing) == len(ids):
        return 0
    if existing:
        store.delete(ids=existing)
    for i in range(0, len(chunks), batch):
        store.add_documents(chunks[i:i + batch], ids=ids[i:i + batch])
        if progress:
            progress(min(i + batch, len(chunks)), len(chunks))
    return len(chunks)


def check_upload(path, max_pages: int) -> tuple[int, str | None]:
    """Page count of an uploaded PDF and, if it cannot be used, the reason why."""
    try:
        with fitz.open(path) as pdf:
            n = len(pdf)
            with_text = sum(len(page.get_text().strip()) >= 100 for page in pdf)
    except Exception:
        return 0, "This file could not be opened as a PDF."
    if n > max_pages:
        return n, f"This PDF has {n} pages; the demo indexes up to {max_pages}."
    if with_text < max(1, n // 3):
        return n, ("This PDF has almost no extractable text (it looks scanned). Only text PDFs, like the "
                   "filings on sec.gov, are supported.")
    return n, None
