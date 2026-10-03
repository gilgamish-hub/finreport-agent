"""MCP server tests through an in-memory MCP client: no index, no LLM."""

import asyncio
import json

from langchain_core.documents import Document
from mcp import Client

from finagent import mcp_server

CASHFLOW = Document(page_content="[3M_2018_10K | page 60]\nPurchases of property, plant and equipment (PP&E) | (1,577)",
                    metadata={"doc_name": "3M_2018_10K", "page": 59, "chunk_id": "cf"})


def run(coro):
    return asyncio.run(coro)


def text(result) -> str:
    return result.content[0].text


async def call(name, args):
    async with Client(mcp_server.server) as client:
        return await client.call_tool(name, args)


def fake_index(monkeypatch):
    import finagent.retriever as retriever

    mcp_server._docs.cache_clear()
    monkeypatch.setattr(mcp_server, "_docs", lambda: frozenset({"3M_2018_10K"}))
    monkeypatch.setattr(retriever, "search", lambda query, doc_name, k=5: [CASHFLOW])


def test_lists_all_five_tools():
    async def names():
        async with Client(mcp_server.server) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert run(names()) == {"list_reports", "search_report", "read_page", "calculate", "ask_report"}


def test_calculate_is_exact_and_safe():
    assert text(run(call("calculate", {"expression": "(1577 - 1373) / 1373 * 100"}))) == "14.858"
    assert text(run(call("calculate", {"expression": "__import__('os').system('dir')"}))).startswith("error")


def test_search_returns_1_indexed_pages(monkeypatch):
    fake_index(monkeypatch)
    result = run(call("search_report", {"doc_name": "3M_2018_10K", "query": "capital expenditure"}))
    hits = result.structured_content["result"] if result.structured_content else json.loads(text(result))
    assert hits[0]["page"] == 60 and "1,577" in hits[0]["text"]


def test_unknown_report_is_an_error(monkeypatch):
    fake_index(monkeypatch)
    result = run(call("search_report", {"doc_name": "NOPE_2020_10K", "query": "revenue"}))
    assert result.is_error and "list_reports" in text(result)
