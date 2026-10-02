"""Routing tests with a scripted LLM and a fake retriever: no API calls, no vector store."""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from finagent import config
from finagent.graph import Grade, Plan, Rewrite, Verdict, build_graph, run_agent

CASHFLOW = Document(
    page_content="[3M_2018_10K | page 60]\nPurchases of property, plant and equipment (PP&E) | (1,577) | (1,373)",
    metadata={"doc_name": "3M_2018_10K", "page": 59, "chunk_id": "cf"},
)
NOISE = Document(page_content="[3M_2018_10K | page 3]\nTable of contents", metadata={
    "doc_name": "3M_2018_10K", "page": 2, "chunk_id": "toc"})


class ScriptedLLM:
    """Returns pre-written structured outputs and chat replies in order."""

    def __init__(self, structured=(), replies=()):
        self.structured = list(structured)
        self.replies = list(replies)

    def _next_structured(self, schema):
        out = self.structured.pop(0)
        assert isinstance(out, schema), f"expected {schema.__name__}, script has {type(out).__name__}"
        return out

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: self._next_structured(schema))

    def bind_tools(self, tools):
        return RunnableLambda(lambda _: self.replies.pop(0))


def call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


def fake_search(results_per_round):
    rounds = iter(results_per_round)
    return lambda queries, doc_name: next(rounds)


def test_lookup_with_calculator_and_grounded_answer():
    fast = ScriptedLLM(structured=[
        Plan(qtype="calculate", queries=["capex 2018", "capex 2017"]),
        Grade(relevant=[0], sufficient=True),
    ])
    main = ScriptedLLM(replies=[
        AIMessage("", tool_calls=[call("calculator", {"expression": "(1577-1373)/1373*100"}, "c1")]),
        AIMessage("", tool_calls=[call("submit_answer", {"answer": "Capex rose 14.86% to $1,577 million.",
                                                          "pages": [60]}, "s1")]),
    ])
    graph = build_graph(main, fast, search_fn=fake_search([[CASHFLOW, NOISE]]))
    out = run_agent(graph, "How much did 3M's capex grow in FY2018?", "3M_2018_10K")

    assert out["status"] == "answered"
    assert out["pages"] == [59]
    assert out["calcs"] == ["(1577-1373)/1373*100 = 14.858"]
    assert [t["step"] for t in out["trace"]] == ["classify", "retrieve", "grade", "reason", "calculator",
                                                 "reason", "verify"]
    assert not main.replies and not fast.structured


def test_rewrites_query_when_first_search_misses():
    fast = ScriptedLLM(structured=[
        Plan(qtype="lookup", queries=["capital expenditure"]),
        Grade(relevant=[], sufficient=False, missing="capex figure"),
        Rewrite(queries=["purchases of property, plant and equipment"]),
        Grade(relevant=[0], sufficient=True),
    ])
    main = ScriptedLLM(replies=[AIMessage("", tool_calls=[
        call("submit_answer", {"answer": "$1,577 million", "pages": [60]}, "s1")])])
    graph = build_graph(main, fast, search_fn=fake_search([[NOISE], [CASHFLOW]]))
    out = run_agent(graph, "What was 3M's FY2018 capex?", "3M_2018_10K")

    assert out["status"] == "answered"
    assert "rewrite" in [t["step"] for t in out["trace"]]


def test_gives_up_when_nothing_relevant_is_found():
    no = Grade(relevant=[], sufficient=False, missing="everything")
    fast = ScriptedLLM(structured=[
        Plan(qtype="lookup", queries=["q"]), no,
        Rewrite(queries=["q2"]), no,
        Rewrite(queries=["q3"]), no,
    ])
    graph = build_graph(ScriptedLLM(), fast, search_fn=fake_search([[NOISE]] * config.MAX_RETRIEVAL_ROUNDS))
    out = run_agent(graph, "What is the CEO's favourite colour?", "3M_2018_10K")

    assert out["status"] == "not_found"
    assert out["answer"] == config.NOT_FOUND


def test_unsupported_number_is_retried_then_refused():
    fast = ScriptedLLM(structured=[
        Plan(qtype="lookup", queries=["capex"]),
        Grade(relevant=[0], sufficient=True),
        Verdict(supported=False, problem="2,000 is not in the evidence"),
        Verdict(supported=False, problem="still wrong"),
    ])
    wrong = {"answer": "$2,000 million", "pages": [60]}
    main = ScriptedLLM(replies=[
        AIMessage("", tool_calls=[call("submit_answer", wrong, "s1")]),
        AIMessage("", tool_calls=[call("submit_answer", wrong, "s2")]),
    ])
    graph = build_graph(main, fast, search_fn=fake_search([[CASHFLOW]]))
    out = run_agent(graph, "What was 3M's FY2018 capex?", "3M_2018_10K")

    assert out["status"] == "not_found"
    assert [t["step"] for t in out["trace"]].count("verify") == 2


def test_plain_text_answer_takes_pages_from_text():
    fast = ScriptedLLM(structured=[Plan(qtype="lookup", queries=["capex"]), Grade(relevant=[0, 1], sufficient=True)])
    main = ScriptedLLM(replies=[AIMessage("Capex was $1,577 million (cash flow statement, page 60).")])
    graph = build_graph(main, fast, search_fn=fake_search([[CASHFLOW, NOISE]]))
    out = run_agent(graph, "What was 3M's FY2018 capex?", "3M_2018_10K")

    assert out["status"] == "answered"
    assert out["pages"] == [59]


def test_partial_answer_is_not_treated_as_refusal():
    fast = ScriptedLLM(structured=[Plan(qtype="lookup", queries=["capex"]), Grade(relevant=[0], sufficient=True)])
    partial = "FY2018 capex was $1,577 million (page 60). The excerpts do not show FY2016, so I could not find it."
    main = ScriptedLLM(replies=[AIMessage("", tool_calls=[call("submit_answer", {"answer": partial, "pages": [60]}, "s1")])])
    graph = build_graph(main, fast, search_fn=fake_search([[CASHFLOW]]))
    out = run_agent(graph, "What was 3M's capex in FY2018 and FY2016?", "3M_2018_10K")

    assert out["status"] == "answered"


def test_reasoning_sees_expanded_pages():
    full_page = Document(page_content="[3M_2018_10K | page 60]\nFULL PAGE", metadata={"doc_name": "3M_2018_10K", "page": 59,
                                                                                     "chunk_id": "3M:59"})
    seen = []
    fast = ScriptedLLM(structured=[Plan(qtype="lookup", queries=["capex"]), Grade(relevant=[0], sufficient=True)])
    main = ScriptedLLM()
    main.bind_tools = lambda tools: RunnableLambda(lambda msgs: seen.append(msgs[-1].content) or AIMessage(
        "", tool_calls=[call("submit_answer", {"answer": "See page.", "pages": [60]}, "s1")]))
    graph = build_graph(main, fast, search_fn=fake_search([[CASHFLOW]]), expand_fn=lambda chunks: [full_page])
    out = run_agent(graph, "What was 3M's FY2018 capex?", "3M_2018_10K")

    assert "FULL PAGE" in seen[0]
    assert out["pages"] == [59]
