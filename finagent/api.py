"""HTTP API for the agent, so other software can use it (the Streamlit app is for people).

    uvicorn finagent.api:app --reload        then open http://localhost:8000/docs

GET  /health         is the service up, how many reports are indexed
GET  /reports        report names you can ask about
POST /ask            question in, answer + cited pages out
POST /ask/stream     same, but each agent step is sent as it happens (server-sent events)
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finagent import config
from finagent.demo import ensure_index
from finagent.graph import build_graph, run_agent
from finagent.ingest import indexed_docs
from finagent.llm import get_llm

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once when the server starts, not on every request: loading models and the index is slow.
    ensure_index()
    state["docs"] = indexed_docs()
    state["graph"] = build_graph(get_llm(config.MAIN_LLM), get_llm(config.FAST_LLM))
    yield
    state.clear()


app = FastAPI(title="FinReport Agent API", version="2.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500, examples=["What was 3M's capital expenditure in FY2018?"])
    doc_name: str = Field(examples=["3M_2018_10K"])


class AskResponse(BaseModel):
    answer: str
    status: str                 # answered | not_found
    pages: list[int]            # 1-indexed report pages the answer cites
    question_type: str | None
    calculations: list[str]
    seconds: float


def check_doc(doc_name: str):
    if doc_name not in state["docs"]:
        raise HTTPException(status_code=404, detail=f"Report '{doc_name}' is not indexed. See GET /reports.")


@app.get("/health")
def health():
    return {"status": "ok", "reports_indexed": len(state.get("docs", []))}


@app.get("/reports")
def reports() -> list[str]:
    return sorted(state["docs"])


# A plain `def` endpoint: FastAPI runs it in a worker thread, so a slow agent run doesn't block other requests.
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    check_doc(req.doc_name)
    start = time.time()
    result = run_agent(state["graph"], req.question, req.doc_name)
    return AskResponse(
        answer=result["answer"],
        status=result["status"],
        pages=[p + 1 for p in result["pages"]],
        question_type=result["qtype"],
        calculations=result["calcs"],
        seconds=round(time.time() - start, 1),
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    check_doc(req.doc_name)

    def events():
        # stream_mode="updates" yields {node_name: what_that_node_changed} after every step of the graph.
        for update in state["graph"].stream({"question": req.question, "doc_name": req.doc_name},
                                            config={"recursion_limit": 50}, stream_mode="updates"):
            for node, changes in update.items():
                for step in (changes or {}).get("trace", []):
                    yield f"data: {json.dumps({'event': 'step', 'node': node, **step})}\n\n"
                if changes and changes.get("status") in ("answered", "not_found"):
                    final = {"event": "answer", "answer": changes["answer"], "status": changes["status"],
                             "pages": [p + 1 for p in changes.get("pages", [])]}
                    yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
