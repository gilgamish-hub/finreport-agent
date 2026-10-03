"""MCP server: lets any MCP client (Claude Desktop, Claude Code, Cursor, other agents) use FinReport.

    python -m finagent.mcp_server                       # stdio, for desktop clients
    python -m finagent.mcp_server --http --port 8001    # streamable HTTP at /mcp

Tools
    list_reports      report names in the index
    search_report     hybrid search inside one report, returns passages with page numbers
    read_page         full text of one page
    calculate         safe arithmetic (same AST calculator the agent uses)
    ask_report        run the full agent: answer + cited pages

The first four need no LLM key: a client's own model can do the reasoning with them.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # message is shown to the client (other errors are hidden)

server = MCPServer(
    name="finreport",
    instructions=("Tools for answering questions about company filings (10-K, 10-Q, earnings releases). "
                  "Call list_reports first, then search_report and read_page to find evidence, and calculate "
                  "for any arithmetic. Cite page numbers. ask_report runs the whole FinReport agent instead."),
)


@lru_cache(maxsize=1)
def _docs() -> frozenset[str]:
    from finagent.ingest import indexed_docs

    return frozenset(indexed_docs())


def _check(doc_name: str):
    if doc_name not in _docs():
        raise ToolError(f"Report '{doc_name}' is not indexed. Call list_reports for valid names.")


@server.tool()
def list_reports() -> list[str]:
    """Names of all indexed company filings, e.g. 3M_2018_10K or AMD_2022_10K."""
    return sorted(_docs())


@server.tool()
def search_report(doc_name: str, query: str, k: int = 5) -> list[dict]:
    """Hybrid (semantic + keyword) search inside one filing.

    Use the report's own wording, e.g. "purchases of property, plant and equipment" rather than "capex".
    Returns up to k passages with their 1-indexed page numbers.
    """
    from finagent.retriever import search

    _check(doc_name)
    hits = search(query, doc_name, k=max(1, min(k, 10)))
    return [{"page": h.metadata["page"] + 1, "text": h.page_content} for h in hits]


@server.tool()
def read_page(doc_name: str, page: int) -> str:
    """Full text of one page (1-indexed) of a filing, with table rows kept on one line."""
    from finagent.retriever import page_text

    _check(doc_name)
    text = page_text(doc_name, page - 1)
    if text.strip().endswith("]"):          # only the header: no chunks on that page
        raise ToolError(f"Page {page} of {doc_name} has no indexed text.")
    return text


@server.tool()
def calculate(expression: str) -> str:
    """Evaluate arithmetic exactly, e.g. "(1577 - 1373) / 1373 * 100". Supports + - * / ** %, abs, round, min, max, sqrt."""
    from finagent.tools import calculator

    return calculator.invoke({"expression": expression})


@lru_cache(maxsize=2)
def _graph(mode: str):
    from finagent import config
    from finagent.graph import build_graph
    from finagent.llm import get_llm

    return build_graph(get_llm(config.MAIN_LLM), get_llm(config.FAST_LLM), route_lookups=(mode == "auto"))


@server.tool()
def ask_report(question: str, doc_name: str, mode: Literal["auto", "agent"] = "auto") -> dict:
    """Run the full FinReport agent on one filing and return its verified answer with cited pages.

    Needs an LLM key on the server (GROQ_API_KEY or GOOGLE_API_KEY). Takes 10-60 seconds.
    """
    from finagent.graph import run_agent

    _check(doc_name)
    res = run_agent(_graph(mode), question, doc_name)
    return {"answer": res["answer"], "status": res["status"], "pages": [p + 1 for p in res["pages"]],
            "route": res["route"], "calculations": res["calcs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    if args.http:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
