"""The LangGraph agent.

    classify --(route_lookups and a plain lookup)--> rag (one search, one answer) -> END
    classify -> retrieve -> grade --(not enough, rounds left)--> rewrite -> retrieve ...
                              |--(enough)--> reason <-> calculator
                              |--(nothing found)--> not_found
    reason -> verify --(grounded)--> END
                     --(unsupported, retry left)--> reason
                     --(unsupported again)--> not_found
"""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from finagent import config, prompts
from finagent.grounding import ungrounded_numbers
from finagent.llm import structured
from finagent.tools import calculator, submit_answer


# ---------- structured outputs ----------

class Plan(BaseModel):
    qtype: Literal["lookup", "calculate", "compare"]
    queries: list[str] = Field(description="1-4 search queries, one per figure or fact needed")


class Grade(BaseModel):
    relevant: list[int] = Field(description="indices of new passages that contain needed information")
    sufficient: bool
    missing: str = Field(default="", description="what is still missing, if not sufficient")


class Rewrite(BaseModel):
    queries: list[str]


class Verdict(BaseModel):
    supported: bool
    problem: str = ""


# ---------- state ----------

class AgentState(TypedDict, total=False):
    question: str
    doc_name: str
    qtype: str
    queries: list[str]
    tried: list[str]
    candidates: list[Document]
    kept: list[Document]
    evidence: list[Document]  # full pages behind the kept chunks, given to reason/verify
    missing: str
    rounds: int
    messages: Annotated[list, add_messages]
    reason_steps: int
    calcs: Annotated[list[str], operator.add]
    verify_retries: int
    answer: str
    pages: list[int]          # 0-indexed pages the answer cites
    status: str               # answered | not_found
    trace: Annotated[list[dict], operator.add]


def format_chunks(chunks: list[Document], numbered: bool = False) -> str:
    parts = []
    for i, c in enumerate(chunks):
        prefix = f"<{i}> " if numbered else ""
        parts.append(prefix + c.page_content)
    return "\n\n---\n\n".join(parts)


def _is_not_found(answer: str) -> bool:
    # Only a full refusal counts; a partial answer may legitimately say "the report does not show FY2021".
    a = answer.strip().lower().replace("’", "'")
    return not a or a.startswith(config.NOT_FOUND.lower().rstrip("."))


def build_graph(main_llm, fast_llm, search_fn=None, expand_fn=None, use_calculator: bool = True,
                verify: bool = True, route_lookups: bool = False, rag_search_fn=None):
    """Compile the agent.

    search_fn(queries, doc_name) -> chunks   defaults to hybrid retrieval
    expand_fn(chunks) -> pages               defaults to full-page expansion (identity if search_fn is given)
    route_lookups                            send questions classified as a plain lookup to plain RAG
                                             (one search, one LLM call) instead of the full loop
    rag_search_fn(query, doc_name, k)        search used by that RAG path (defaults to hybrid search)
    """
    if search_fn is None:
        from finagent.retriever import expand_to_pages, search_many
        search_fn, expand_fn = search_many, expand_fn or expand_to_pages
    expand_fn = expand_fn or (lambda chunks: chunks)

    planner = structured(fast_llm, Plan)
    grader = structured(fast_llm, Grade)
    rewriter = structured(fast_llm, Rewrite)
    verifier = structured(fast_llm, Verdict)
    tools = [calculator, submit_answer] if use_calculator else [submit_answer]
    reasoner = main_llm.bind_tools(tools).with_retry(stop_after_attempt=3)

    def classify(s: AgentState):
        plan = planner.invoke(prompts.CLASSIFY.format(doc_name=s["doc_name"], question=s["question"]))
        queries = [q for q in plan.queries if q.strip()][:4] or [s["question"]]
        return {"qtype": plan.qtype, "queries": queries, "tried": [], "kept": [], "rounds": 0,
                "reason_steps": 0, "verify_retries": 0,
                "trace": [{"step": "classify", "type": plan.qtype, "queries": queries}]}

    def retrieve(s: AgentState):
        kept_ids = {c.metadata["chunk_id"] for c in s.get("kept", [])}
        found = search_fn(s["queries"], s["doc_name"])
        new = [c for c in found if c.metadata["chunk_id"] not in kept_ids][: config.MAX_CONTEXT_CHUNKS]
        return {"candidates": new, "rounds": s.get("rounds", 0) + 1,
                "tried": s.get("tried", []) + s["queries"],
                "trace": [{"step": "retrieve", "queries": s["queries"],
                           "pages": sorted({c.metadata["page"] + 1 for c in new})}]}

    def grade(s: AgentState):
        kept = s.get("kept", [])
        kept_txt = f"Passages already kept:\n{format_chunks(kept)}\n" if kept else ""
        g = grader.invoke(prompts.GRADE.format(
            doc_name=s["doc_name"], question=s["question"], kept=kept_txt,
            candidates=format_chunks(s["candidates"], numbered=True)))
        picked = [s["candidates"][i] for i in dict.fromkeys(g.relevant) if 0 <= i < len(s["candidates"])]
        kept = (kept + picked)[: config.MAX_KEPT_CHUNKS]
        return {"kept": kept, "missing": "" if g.sufficient else g.missing,
                "trace": [{"step": "grade", "relevant_pages": sorted({c.metadata["page"] + 1 for c in picked}),
                           "sufficient": g.sufficient, "missing": g.missing}]}

    def after_classify(s: AgentState) -> str:
        return "rag" if route_lookups and s.get("qtype") == "lookup" else "retrieve"

    def rag(s: AgentState):
        # The evaluation showed the full loop pays off on calculations and comparisons, while plain RAG
        # is as good or better on single-fact lookups at a third of the cost.
        from finagent.baseline import run_baseline   # local import: baseline imports this module

        res = run_baseline(main_llm, s["question"], s["doc_name"], search_fn=rag_search_fn)
        return {"answer": res["answer"], "pages": res["pages"], "status": res["status"],
                "trace": [{"step": "rag", "pages": [p + 1 for p in res["retrieved_pages"]]}]}

    def after_grade(s: AgentState) -> str:
        if s["kept"] and not s["missing"]:
            return "reason"
        if s["rounds"] < config.MAX_RETRIEVAL_ROUNDS:
            return "rewrite"
        return "reason" if s["kept"] else "not_found"

    def rewrite(s: AgentState):
        r = rewriter.invoke(prompts.REWRITE.format(
            doc_name=s["doc_name"], question=s["question"], tried="; ".join(s["tried"]),
            missing=s["missing"] or "the passages found were not relevant"))
        queries = [q for q in r.queries if q.strip()][:3] or [s["question"]]
        return {"queries": queries, "trace": [{"step": "rewrite", "queries": queries}]}

    def reason(s: AgentState):
        msgs, update = [], {}
        if not s.get("messages"):
            evidence = sorted(expand_fn(s["kept"]), key=lambda d: d.metadata["page"])
            update["evidence"] = evidence
            calc_rule = prompts.CALC_RULE_TOOL if use_calculator else prompts.CALC_RULE_NO_TOOL
            msgs = [
                SystemMessage(prompts.REASON_SYSTEM.format(
                    doc_name=s["doc_name"], calc_rule=calc_rule, not_found=config.NOT_FOUND)),
                HumanMessage(prompts.REASON_USER.format(
                    evidence=format_chunks(evidence), question=s["question"])),
            ]
        response = reasoner.invoke(list(s.get("messages", [])) + msgs)
        calls = [c["name"] for c in getattr(response, "tool_calls", [])]
        return {**update, "messages": msgs + [response], "reason_steps": s.get("reason_steps", 0) + 1,
                "trace": [{"step": "reason", "tool_calls": calls}]}

    def after_reason(s: AgentState) -> str:
        last = s["messages"][-1]
        names = [c["name"] for c in getattr(last, "tool_calls", [])]
        if "calculator" in names and "submit_answer" not in names and s["reason_steps"] < config.MAX_REASON_STEPS:
            return "calculator"
        return "verify"

    def run_calculator(s: AgentState):
        out, calcs = [], []
        for call in s["messages"][-1].tool_calls:
            result = calculator.invoke(call["args"]) if call["name"] == "calculator" else "not run"
            out.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
            if call["name"] == "calculator":
                calcs.append(f"{call['args'].get('expression', '')} = {result}")
        return {"messages": out, "calcs": calcs, "trace": [{"step": "calculator", "results": calcs}]}

    def check(s: AgentState):
        last: AIMessage = s["messages"][-1]
        submit = next((c for c in getattr(last, "tool_calls", []) if c["name"] == "submit_answer"), None)
        if submit:
            answer = str(submit["args"].get("answer", "")).strip()
            raw_pages = submit["args"].get("pages") or []
        else:
            answer = last.text if hasattr(last, "text") else str(last.content)
            raw_pages = re.findall(r"\bpages?\s+(\d+)", answer, flags=re.I)   # "(page 60)" in a plain-text answer
        pages = []
        for p in raw_pages:
            try:
                pages.append(int(p) - 1)       # model sees 1-indexed pages
            except (TypeError, ValueError):
                pass
        kept_pages = {c.metadata["page"] for c in s["evidence"]}
        pages = sorted({p for p in pages if p in kept_pages}) or sorted(kept_pages)

        if _is_not_found(answer):
            return {"answer": config.NOT_FOUND, "pages": [], "status": "not_found",
                    "trace": [{"step": "verify", "result": "model reported not found"}]}

        evidence = format_chunks(s["evidence"])
        missing = ungrounded_numbers(answer, evidence, s.get("calcs", []), s["question"])
        if not missing or not verify:
            return {"answer": answer, "pages": pages, "status": "answered",
                    "trace": [{"step": "verify", "result": "all numbers grounded" if not missing
                               else "verification disabled"}]}

        v = verifier.invoke(prompts.VERIFY.format(
            doc_name=s["doc_name"], question=s["question"], answer=answer,
            calcs="; ".join(s.get("calcs", [])) or "none", evidence=evidence))
        if v.supported:
            return {"answer": answer, "pages": pages, "status": "answered",
                    "trace": [{"step": "verify", "result": f"LLM check passed (unmatched numbers {missing})"}]}

        if s.get("verify_retries", 0) < config.MAX_VERIFY_RETRIES:
            feedback = (f"Your answer was rejected by verification: {v.problem} "
                        f"Numbers not found in the evidence or calculations: {missing}. "
                        "Re-check the evidence, use the calculator for any arithmetic, and submit again.")
            # Every tool call in the last turn needs a reply before the model can continue.
            replies = []
            for call in getattr(last, "tool_calls", []):
                content = feedback if call["name"] == "submit_answer" else calculator.invoke(call["args"])
                replies.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            if not submit:
                replies.append(HumanMessage(feedback))
            return {"messages": replies, "verify_retries": s.get("verify_retries", 0) + 1,
                    "status": "retry", "trace": [{"step": "verify", "result": f"rejected: {v.problem}"}]}

        return {"answer": config.NOT_FOUND, "pages": [], "status": "not_found",
                "trace": [{"step": "verify", "result": f"rejected twice: {v.problem}"}]}

    def after_verify(s: AgentState) -> str:
        return "reason" if s["status"] == "retry" else END

    def not_found(s: AgentState):
        return {"answer": config.NOT_FOUND, "pages": [], "status": "not_found",
                "trace": [{"step": "not_found"}]}

    g = StateGraph(AgentState)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("reason", reason)
    g.add_node("calculator", run_calculator)
    g.add_node("verify", check)
    g.add_node("not_found", not_found)
    g.add_node("rag", rag)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", after_classify, ["retrieve", "rag"])
    g.add_edge("rag", END)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["reason", "rewrite", "not_found"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("reason", after_reason, ["calculator", "verify"])
    g.add_edge("calculator", "reason")
    g.add_conditional_edges("verify", after_verify, ["reason", END])
    g.add_edge("not_found", END)
    return g.compile()


def run_agent(graph, question: str, doc_name: str, callbacks=None) -> dict:
    final = graph.invoke({"question": question, "doc_name": doc_name},
                         config={"callbacks": callbacks or [], "recursion_limit": 50})
    return {
        "answer": final.get("answer", config.NOT_FOUND),
        "status": final.get("status", "not_found"),
        "pages": final.get("pages", []),
        "qtype": final.get("qtype"),
        "route": "rag" if any(t.get("step") == "rag" for t in final.get("trace", [])) else "agent",
        "calcs": final.get("calcs", []),
        "retrieved_pages": sorted({c.metadata["page"] for c in final.get("kept", [])}),
        "trace": final.get("trace", []),
        "evidence": final.get("evidence", []),
    }


def trace_to_json(trace: list[dict]) -> str:
    return json.dumps(trace, ensure_ascii=False)
