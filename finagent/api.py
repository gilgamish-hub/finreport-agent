"""HTTP API for the agent, so other software can use it (the Streamlit app is for people).

    uvicorn finagent.api:app --reload        then open http://localhost:8000/docs

GET  /health         is the service up, how many reports are indexed
GET  /reports        report names you can ask about
POST /ask            question in, answer + cited pages out
POST /ask/stream     same, but each agent step is sent as it happens (server-sent events)
GET  /metrics        requests, routes, errors, latency, LLM calls and tokens since start

Set FINAGENT_API_KEY to require an "X-API-Key" header on every endpoint except /health.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finagent import config
from finagent.demo import ensure_index, is_rate_limit
from finagent.graph import build_graph, run_agent
from finagent.ingest import indexed_docs
from finagent.llm import CallCounter, get_llm

log = logging.getLogger("finagent.api")
state: dict = {}


class Metrics:
    """In-process counters. Enough for one instance; several instances would export to Prometheus instead."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        self.by_route: dict[str, int] = {}
        self.by_status: dict[str, int] = {}
        self.seconds = 0.0
        self.max_seconds = 0.0
        self.llm_calls = 0
        self.tokens = 0

    def record(self, route: str, status: str, seconds: float, calls: int, tokens: int):
        with self.lock:
            self.requests += 1
            self.by_route[route] = self.by_route.get(route, 0) + 1
            self.by_status[status] = self.by_status.get(status, 0) + 1
            self.seconds += seconds
            self.max_seconds = max(self.max_seconds, seconds)
            self.llm_calls += calls
            self.tokens += tokens

    def error(self):
        with self.lock:
            self.errors += 1

    def snapshot(self) -> dict:
        with self.lock:
            n = self.requests or 1
            return {"uptime_seconds": round(time.time() - self.started), "requests": self.requests,
                    "errors": self.errors, "by_route": dict(self.by_route), "by_status": dict(self.by_status),
                    "avg_seconds": round(self.seconds / n, 1), "max_seconds": round(self.max_seconds, 1),
                    "avg_llm_calls": round(self.llm_calls / n, 2), "avg_tokens": round(self.tokens / n),
                    "total_tokens": self.tokens}


metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once when the server starts, not on every request: loading models and the index is slow.
    ensure_index()
    state["docs"] = indexed_docs()
    main, fast = get_llm(config.MAIN_LLM), get_llm(config.FAST_LLM)
    # "auto" sends single-fact lookups to plain RAG and everything else through the full agent loop
    state["graphs"] = {"auto": build_graph(main, fast, route_lookups=True), "agent": build_graph(main, fast)}
    yield
    state.clear()


app = FastAPI(title="FinReport Agent API", version="2.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("FINAGENT_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500, examples=["What was 3M's capital expenditure in FY2018?"])
    doc_name: str = Field(examples=["3M_2018_10K"])
    mode: Literal["auto", "agent"] = Field(
        default="auto", description="auto: lookups go to plain RAG, the rest to the agent; agent: always the full loop")


class AskResponse(BaseModel):
    answer: str
    status: str                 # answered | not_found
    pages: list[int]            # 1-indexed report pages the answer cites
    question_type: str | None
    route: str                  # rag | agent
    calculations: list[str]
    seconds: float
    llm_calls: int
    tokens: int


def check_doc(doc_name: str):
    if doc_name not in state["docs"]:
        raise HTTPException(status_code=404, detail=f"Report '{doc_name}' is not indexed. See GET /reports.")


def log_request(req: AskRequest, route: str, status: str, seconds: float, counter: CallCounter):
    metrics.record(route, status, seconds, counter.calls, counter.tokens)
    log.info(json.dumps({"event": "ask", "doc": req.doc_name, "mode": req.mode, "route": route, "status": status,
                         "seconds": round(seconds, 1), "llm_calls": counter.calls, "tokens": counter.tokens}))


@app.get("/health")
def health():
    return {"status": "ok", "reports_indexed": len(state.get("docs", []))}


@app.get("/reports", dependencies=[Depends(require_key)])
def reports() -> list[str]:
    return sorted(state["docs"])


@app.get("/metrics", dependencies=[Depends(require_key)])
def get_metrics():
    return metrics.snapshot()


# A plain `def` endpoint: FastAPI runs it in a worker thread, so a slow agent run doesn't block other requests.
@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
def ask(req: AskRequest):
    check_doc(req.doc_name)
    counter, start = CallCounter(), time.time()
    try:
        result = run_agent(state["graphs"][req.mode], req.question, req.doc_name, callbacks=[counter])
    except Exception as e:
        metrics.error()
        log.exception("ask failed")
        if is_rate_limit(e):
            raise HTTPException(status_code=503, detail="The model's rate limit is reached. Try again in a minute.")
        raise HTTPException(status_code=500, detail="The agent failed on this question.")
    seconds = time.time() - start
    log_request(req, result["route"], result["status"], seconds, counter)
    return AskResponse(
        answer=result["answer"],
        status=result["status"],
        pages=[p + 1 for p in result["pages"]],
        question_type=result["qtype"],
        route=result["route"],
        calculations=result["calcs"],
        seconds=round(seconds, 1),
        llm_calls=counter.calls,
        tokens=counter.tokens,
    )


@app.post("/ask/stream", dependencies=[Depends(require_key)])
def ask_stream(req: AskRequest):
    check_doc(req.doc_name)

    def events():
        counter, start, route = CallCounter(), time.time(), "agent"
        try:
            # stream_mode="updates" yields {node_name: what_that_node_changed} after every step of the graph.
            for update in state["graphs"][req.mode].stream(
                    {"question": req.question, "doc_name": req.doc_name},
                    config={"recursion_limit": 50, "callbacks": [counter]}, stream_mode="updates"):
                for node, changes in update.items():
                    route = "rag" if node == "rag" else route
                    for step in (changes or {}).get("trace", []):
                        yield f"data: {json.dumps({'event': 'step', 'node': node, **step})}\n\n"
                    if changes and changes.get("status") in ("answered", "not_found"):
                        seconds = time.time() - start
                        log_request(req, route, changes["status"], seconds, counter)
                        final = {"event": "answer", "answer": changes["answer"], "status": changes["status"],
                                 "pages": [p + 1 for p in changes.get("pages", [])], "route": route,
                                 "seconds": round(seconds, 1), "llm_calls": counter.calls, "tokens": counter.tokens}
                        yield f"data: {json.dumps(final)}\n\n"
        except Exception as e:
            metrics.error()
            log.exception("ask/stream failed")
            msg = "rate limit reached, try again in a minute" if is_rate_limit(e) else "the agent failed"
            yield f"data: {json.dumps({'event': 'error', 'detail': msg})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
