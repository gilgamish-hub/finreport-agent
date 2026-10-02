from langchain_core.documents import Document

from finagent.ingest import chunk_pages, clean_page_text
from finagent.retriever import rrf, tokenize


def test_clean_page_text_keeps_table_columns():
    raw = ("(Millions)                         2018          2017\n"
           "Net income                    $      5,363   $   4,869\n\n\n")
    assert clean_page_text(raw) == "(Millions) | 2018 | 2017\nNet income | $5,363 | $4,869"


def test_chunks_keep_page_and_get_header():
    pages = [Document(page_content="word " * 800, metadata={"doc_name": "X_2020_10K", "company": "X", "page": 4})]
    chunks = chunk_pages(pages)
    assert len(chunks) > 1
    assert all(c.metadata["page"] == 4 for c in chunks)
    assert chunks[0].page_content.startswith("[X_2020_10K | page 5]")
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_rrf_rewards_agreement():
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


def test_tokenize_drops_stopwords():
    assert tokenize("What is the FY2018 capex?") == ["fy2018", "capex"]


def test_join_overlapping_removes_chunk_overlap():
    from finagent.retriever import _join_overlapping

    assert _join_overlapping("Net sales | 100\nCost of sales | 60", "Cost of sales | 60\nGross profit | 40") == \
        "Net sales | 100\nCost of sales | 60\nGross profit | 40"
    assert _join_overlapping("abc", "xyz") == "abc\nxyz"


def test_check_upload_rejects_scanned_and_long_pdfs(tmp_path):
    import fitz

    from finagent.ingest import check_upload

    blank = tmp_path / "scan.pdf"
    with fitz.open() as pdf:
        for _ in range(3):
            pdf.new_page()
        pdf.save(blank)
    assert "scanned" in check_upload(blank, max_pages=150)[1]
    assert "up to 2" in check_upload(blank, max_pages=2)[1]

    text = tmp_path / "text.pdf"
    with fitz.open() as pdf:
        for i in range(3):
            pdf.new_page().insert_text((72, 72), f"Net sales for fiscal 2023 were $1,234 million. Page {i}. " * 3)
        pdf.save(text)
    assert check_upload(text, max_pages=150) == (3, None)
