"""API tests with a fake graph: no LLM calls, no vector store (the startup step is skipped)."""

import json

from fastapi.testclient import TestClient

from finagent import api

FINAL = {"answer": "$1,577 million", "status": "answered", "pages": [59], "qtype": "lookup",
         "calcs": [], "kept": [], "trace": [], "evidence": []}


class FakeGraph:
    def invoke(self, inputs, config=None):
        return FINAL

    def stream(self, inputs, config=None, stream_mode=None):
        yield {"classify": {"trace": [{"step": "classify", "type": "lookup", "queries": ["capex 2018"]}]}}
        yield {"verify": {**FINAL, "trace": [{"step": "verify", "result": "all numbers grounded"}]}}


def client():
    api.state.update(docs={"3M_2018_10K"}, graphs={"auto": FakeGraph(), "agent": FakeGraph()})
    return TestClient(api.app)   # not used as a context manager, so the real startup does not run


def test_ask_returns_answer_with_1_indexed_pages():
    r = client().post("/ask", json={"question": "What was 3M's capex in FY2018?", "doc_name": "3M_2018_10K"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "$1,577 million" and body["pages"] == [60] and body["status"] == "answered"


def test_unknown_report_is_404():
    r = client().post("/ask", json={"question": "What was revenue?", "doc_name": "NOPE_2020_10K"})
    assert r.status_code == 404


def test_too_short_question_is_rejected():
    r = client().post("/ask", json={"question": "hi", "doc_name": "3M_2018_10K"})
    assert r.status_code == 422


def test_stream_sends_steps_then_answer():
    r = client().post("/ask/stream", json={"question": "What was 3M's capex in FY2018?", "doc_name": "3M_2018_10K"})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert [e["event"] for e in events] == ["step", "step", "answer"]
    assert events[-1]["pages"] == [60]


class RateLimitedGraph:
    def invoke(self, inputs, config=None):
        raise RuntimeError("Error code: 429 - rate limit reached")


def test_metrics_count_requests_and_routes():
    api.metrics.__init__()
    c = client()
    c.post("/ask", json={"question": "What was 3M's capex in FY2018?", "doc_name": "3M_2018_10K"})
    m = c.get("/metrics").json()
    assert m["requests"] == 1 and m["by_route"] == {"agent": 1} and m["by_status"] == {"answered": 1}


def test_api_key_is_required_when_set(monkeypatch):
    monkeypatch.setenv("FINAGENT_API_KEY", "secret")
    c = client()
    body = {"question": "What was 3M's capex in FY2018?", "doc_name": "3M_2018_10K"}
    assert c.post("/ask", json=body).status_code == 401
    assert c.post("/ask", json=body, headers={"X-API-Key": "secret"}).status_code == 200
    assert c.get("/health").status_code == 200        # health stays open for load balancers


def test_rate_limit_becomes_503():
    c = client()
    api.state["graphs"]["auto"] = RateLimitedGraph()
    r = c.post("/ask", json={"question": "What was 3M's capex in FY2018?", "doc_name": "3M_2018_10K"})
    assert r.status_code == 503 and "rate limit" in r.json()["detail"]
