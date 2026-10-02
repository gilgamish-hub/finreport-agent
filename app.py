"""Streamlit UI: ask a filing questions and watch the agent work; browse the FinanceBench evaluation."""

import hashlib
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# On Streamlit Cloud, settings arrive as secrets; finagent reads them from the environment at import time.
try:
    for _name, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_name, _value)
except Exception:   # no secrets.toml (local runs use .env)
    pass

from finagent import config  # noqa: E402
from finagent.data import load_questions, load_reports  # noqa: E402
from finagent.demo import (DEMO_CHAIN, AllModelsBusy, AnswerCache, ModelConfig, ensure_index,
                           load_saved_runs, normalize, run_with_fallback)
from finagent.evaluate import load_results, summarize
from finagent.graph import build_graph
from finagent.ingest import check_upload, index_pdf, indexed_docs

st.set_page_config(page_title="Financial Report Analyst", page_icon="📊", layout="wide")

STEP_LABELS = {
    "classify": "Planned the search", "retrieve": "Searched the report", "grade": "Graded the passages",
    "rewrite": "Rewrote the search", "reason": "Reasoned over the evidence", "calculator": "Ran the calculator",
    "verify": "Verified the answer", "not_found": "Gave up",
}

AUTO = "Auto: free models, switches when one is busy"
MODELS = {  # explicit choices besides AUTO
    "Groq · gpt-oss-120b": DEMO_CHAIN[0],
    "Local GPU · qwen3.5 4B (Ollama)": ModelConfig("qwen3.5 4B (local GPU)", "ollama:qwen3.5:4b",
                                                   "ollama:qwen3.5:4b", None),
    "Gemini 3.5 Flash": DEMO_CHAIN[-1],
}
KEY_URLS = {"GROQ_API_KEY": "https://console.groq.com/keys", "GOOGLE_API_KEY": "https://aistudio.google.com/apikey"}
# Live questions per visitor on the shared free keys (0 = no limit). Visitors with their own key are not limited.
LIVE_LIMIT = int(os.getenv("FINAGENT_DEMO_LIMIT", "5"))

RUNS = {  # results file suffix -> description
    "": "Groq · gpt-oss-120b",
    "_ollama": "Local GPU · qwen3.5 4B",
}
VARIANTS = {"baseline": "Plain RAG", "agent_nocalc": "Agent, no calculator", "agent": "Agent"}
LABEL_ICON = {"correct": "✅ correct", "incorrect": "❌ wrong", "refusal": "⏸️ refused"}


@st.cache_data(show_spinner=False)
def reports():
    return load_reports()


@st.cache_data(show_spinner=False)
def questions_by_doc():
    out = {}
    for q in load_questions():
        out.setdefault(q.doc_name, []).append(q)
    return out


@st.cache_data(show_spinner="Loading the vector index...", ttl=600)
def available_docs():
    ensure_index()   # first start on a fresh host: fetch the demo index
    return sorted(indexed_docs())


@st.cache_data(show_spinner=False)
def saved_runs():
    return load_saved_runs()


@st.cache_resource(show_spinner=False)
def answer_cache():
    return AnswerCache()


@st.cache_data(ttl=30, show_spinner=False)
def ollama_running() -> bool:
    try:
        return requests.get("http://localhost:11434/api/version", timeout=1).ok
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner=False)
def agent(api_key, main_model: str, fast_model: str):
    from finagent.llm import get_llm

    return build_graph(get_llm(main_model, api_key), get_llm(fast_model, api_key))


def env_key(name):
    key = os.getenv(name) or (os.getenv("GEMINI_API_KEY") if name == "GOOGLE_API_KEY" else None)
    if not key:
        try:
            key = st.secrets.get(name)
        except Exception:   # no secrets.toml
            key = None
    return key


def describe(step: dict) -> str:
    s = step["step"]
    if s in ("classify", "rewrite"):
        extra = f" ({step['type']})" if s == "classify" else ""
        return f"{STEP_LABELS[s]}{extra}: " + "; ".join(f"`{q}`" for q in step["queries"])
    if s == "retrieve":
        return f"{STEP_LABELS[s]}: pages {', '.join(map(str, step['pages'])) or 'none'}"
    if s == "grade":
        verdict = "enough to answer" if step["sufficient"] else f"missing: {step['missing']}"
        return f"{STEP_LABELS[s]}: kept pages {step['relevant_pages'] or 'none'}, {verdict}"
    if s == "reason":
        return f"{STEP_LABELS[s]}: called {', '.join(step['tool_calls']) or 'no tools'}"
    if s == "calculator":
        return f"{STEP_LABELS[s]}: " + "; ".join(f"`{r}`" for r in step["results"])
    if s == "verify":
        return f"{STEP_LABELS[s]}: {step['result']}"
    return STEP_LABELS.get(s, s)


def label(doc):
    if doc.startswith("UPLOAD_"):
        return f"📄 {st.session_state.get('uploads', {}).get(doc, doc)} (your upload)"
    r = reports().get(doc)
    return f"{r.company} · {r.doc_type.upper()} · {r.period}" if r else doc


def index_upload(upload):
    data = upload.getvalue()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(upload.name).stem).strip("_")[:40] or "report"
    doc_name = f"UPLOAD_{stem}_{hashlib.sha1(data).hexdigest()[:10]}"
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"{doc_name}.pdf"
    path.write_bytes(data)
    pages, problem = check_upload(path, config.MAX_UPLOAD_PAGES)
    if problem:
        st.error(problem)
        return
    bar = st.progress(0.0, text=f"Reading {pages} pages...")
    try:
        index_pdf(path, doc_name, progress=lambda done, total: bar.progress(
            done / total, text=f"Indexing: {done} of {total} passages"), batch=64)
    except Exception as e:
        st.error("Sorry, this PDF could not be indexed.")
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}", language=None)
        return
    st.session_state.setdefault("uploads", {})[doc_name] = upload.name
    available_docs.clear()
    st.session_state.doc_select = doc_name
    st.rerun()


def key_for(cfg: ModelConfig):
    return (own_keys.get(cfg.key_env) or env_key(cfg.key_env)) if cfg.key_env else None


def render_answer(result: dict, note: str, question: str, gold: str | None):
    if note:
        st.caption(note)
    if result.get("trace"):
        with st.expander(f"The agent's steps ({len(result['trace'])})", expanded=False):
            for step in result["trace"]:
                st.markdown("- " + describe(step))
    if result.get("status") == "answered":
        with st.container(border=True):
            st.markdown(result["answer"])
        pages = result.get("pages", [])
        if pages:
            links = [f"[p. {p + 1}]({report.link}#page={p + 1})" if report else f"p. {p + 1}" for p in pages]
            st.markdown("**Sources:** " + " · ".join(links))
        if result.get("calcs"):
            st.markdown("**Calculations:** " + "; ".join(f"`{c}`" for c in result["calcs"]))
        evidence = [e for e in result.get("evidence", []) if e["page"] in pages]
        if evidence:
            with st.expander("Evidence passages"):
                for e in evidence:
                    st.code(e["text"], language=None)
    else:
        st.markdown(f"### {config.NOT_FOUND}")
        st.caption("The agent could not find (or could not verify) the answer in this report.")
    if gold:
        st.info(f"FinanceBench reference answer: {gold}")


def queue_live(question, gold):
    st.session_state.update(pending=question, gold=gold, force_live=True)


def answer_question(question: str, gold: str | None, force_live: bool = False):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        # 1. Free answers first: a saved evaluation run, or an answer given earlier to anyone.
        if not force_live:
            saved = saved_runs().get((doc, normalize(question)))
            cached = answer_cache().get(doc, question)
            if saved or cached:
                if saved:
                    note = (f"⚡ Saved run from the evaluation ({saved['model'].split(':')[-1].split('/')[-1]}, "
                            f"judged {LABEL_ICON[saved['label']]}): shown instantly, uses no API quota.")
                    result = saved
                else:
                    note = f"⚡ Answered earlier by {cached['model']}: shown from the cache, uses no API quota."
                    result = cached
                render_answer(result, note, question, gold)
                st.button("Run it live instead", key=f"live-{hash((doc, question))}",
                          on_click=queue_live, args=(question, gold))
                return

        # 2. Live answer, within the visitor's allowance on the shared free keys.
        own = any(own_keys.values())
        if LIVE_LIMIT and not own and st.session_state.get("live_count", 0) >= LIVE_LIMIT:
            st.info(f"You've used this session's {LIVE_LIMIT} live questions on the shared free API keys. "
                    "Example questions still answer instantly from saved runs, or add your own free Groq key "
                    "in the sidebar for unlimited questions.")
            return
        chain = [c for c in (DEMO_CHAIN if choice == AUTO else [MODELS[choice]]) if not c.key_env or key_for(c)]
        if not chain:
            st.warning("No API key is available for this model. Add one in the sidebar under "
                       "'Use your own API key'.")
            return

        final = {}
        with st.status("Working...", expanded=show_trace) as status:
            def run_one(cfg):
                final.clear()
                final.update(steps=[], calcs=[])
                for update in agent(key_for(cfg), cfg.main, cfg.fast).stream(
                        {"question": question, "doc_name": doc}, stream_mode="updates",
                        config={"recursion_limit": 50}):
                    for out in update.values():
                        final.update({k: v for k, v in out.items() if k not in ("trace", "calcs")})
                        final["calcs"].extend(out.get("calcs", []))
                        for step in out.get("trace", []):
                            final["steps"].append(step)
                            st.markdown("- " + describe(step))
                return final

            def on_switch(busy, nxt):
                st.markdown(f"- ⚠️ **{busy.label}** is at its free limit right now, switching to **{nxt.label}**")

            try:
                _, used = run_with_fallback(chain, run_one, on_switch)
            except AllModelsBusy:
                status.update(label="All free models are busy", state="error")
                st.warning("Every free model this demo uses has reached its limit for now. Free-tier quotas "
                           "reset within 24 hours. Meanwhile, the example questions answer instantly from saved "
                           "runs, or add your own free Groq key in the sidebar.")
                return
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                st.error("Sorry, something went wrong while answering. Please try again or pick an example "
                         "question.")
                with st.expander("Technical details"):
                    st.code(f"{type(e).__name__}: {e}", language=None)
                return
            status.update(label=f"Done in {len(final['steps'])} steps with {used.label}", state="complete",
                          expanded=False)

        result = {
            "answer": final.get("answer", config.NOT_FOUND), "status": final.get("status", "not_found"),
            "pages": final.get("pages", []), "calcs": final["calcs"], "trace": final["steps"],
            "evidence": [{"page": c.metadata["page"], "text": c.page_content} for c in final.get("evidence", [])],
            "model": used.label, "created": time.time(),
        }
        if not own:
            st.session_state.live_count = st.session_state.get("live_count", 0) + 1
        answer_cache().put(doc, question, result)
        render_answer(result, "", question, gold)


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    choices = [AUTO] + [m for m in MODELS if not (MODELS[m].main.startswith("ollama") and not ollama_running())]
    choice = st.selectbox("Model", choices)
    with st.expander("Use your own API key (optional)"):
        st.caption("With your own free key there is no question limit. Keys are used only for this session.")
        own_keys = {name: st.text_input(f"{name.split('_')[0].title()} API key", type="password",
                                        help=f"Free key: {url}") for name, url in KEY_URLS.items()}
    limit_slot = st.empty()   # filled at the end of the script, after this run's question is counted
    show_trace = st.toggle("Show the agent's steps", value=True)

    with st.expander("Upload your own report (PDF)"):
        st.caption(f"A text PDF such as a 10-K or annual report, up to {config.MAX_UPLOAD_PAGES} pages. "
                   "Indexing takes about a minute per 100 pages. Uploads are listed only in your session "
                   "and are not kept.")
        upload = st.file_uploader("PDF report", type="pdf", label_visibility="collapsed")
        if upload is not None and st.button("Index this report", width="stretch"):
            index_upload(upload)

    uploads = st.session_state.get("uploads", {})
    # Other visitors' uploads stay out of the list.
    docs = [d for d in available_docs() if not d.startswith("UPLOAD_") or d in uploads]
    if not docs:
        st.error("No reports are indexed. Run `python scripts/build_index.py`.")
        st.stop()
    if st.session_state.get("doc_select") not in docs:
        st.session_state.doc_select = "3M_2018_10K" if "3M_2018_10K" in docs else docs[0]
    doc = st.selectbox("Report", docs, format_func=label, key="doc_select")
    report = reports().get(doc)
    if report:
        st.markdown(f"[Open the original filing]({report.link})")
    st.caption("Data: FinanceBench (Patronus AI), CC BY-NC 4.0.")

st.title("📊 Financial Report Analyst")
ask_tab, eval_tab = st.tabs(["Ask a report", "Evaluation results"])

# ---------- ask ----------
with ask_tab:
    st.write("A LangGraph agent searches the filing, checks it found the right passages, does the maths "
             "with a calculator tool, verifies the numbers and cites its pages.")

    examples = questions_by_doc().get(doc, [])
    if examples:
        st.caption("FinanceBench questions for this report (answered instantly from the evaluation):")
        cols = st.columns(min(len(examples), 3))
        for i, q in enumerate(examples[:3]):
            if cols[i].button(q.question[:110] + ("…" if len(q.question) > 110 else ""), key=q.id,
                              width="stretch"):
                st.session_state.update(pending=q.question, gold=q.answer, force_live=False)

    question = st.chat_input("e.g. What was the FY2018 capital expenditure?")
    if question:
        st.session_state.update(pending=question, gold=None, force_live=False)

    if st.session_state.get("pending"):
        answer_question(st.session_state.pop("pending"), st.session_state.pop("gold", None),
                        st.session_state.pop("force_live", False))

if LIVE_LIMIT and not any(own_keys.values()):
    left = max(0, LIVE_LIMIT - st.session_state.get("live_count", 0))
    limit_slot.caption(f"Live questions left this session: {left} of {LIVE_LIMIT}. Example questions are free.")

# ---------- evaluation ----------
with eval_tab:
    available = {tag: name for tag, name in RUNS.items()
                 if (config.RESULTS_DIR / f"agent{tag}.jsonl").exists()}
    if not available:
        st.info("No evaluation results yet. Run `python scripts/run_eval.py`.")
        st.stop()
    tag = st.radio("Model", list(available), format_func=available.get, horizontal=True)
    runs = {v: [r for r in load_results(config.RESULTS_DIR / f"{v}{tag}.jsonl") if "label" in r]
            for v in VARIANTS}
    runs = {v: rows for v, rows in runs.items() if rows}
    # Compare on the questions both plain RAG and the agent answered; the no-calculator ablation is
    # shown only where it covers the same questions (on Groq it was run on the first 30 only).
    common = {r["id"] for r in runs["agent"]} & {r["id"] for r in runs.get("baseline", runs["agent"])}
    runs = {v: [r for r in rows if r["id"] in common] for v, rows in runs.items()
            if common <= {r["id"] for r in rows}}
    stats = {v: summarize(rows) for v, rows in runs.items()}

    st.write(f"{len(common)} held-out FinanceBench questions, graded by an LLM judge "
             f"(`{runs['agent'][0].get('judge_model', '-')}`) against the reference answers.")
    cols = st.columns(len(stats))
    for col, (v, s) in zip(cols, stats.items()):
        delta = None
        if v != "baseline" and "baseline" in stats:
            delta = f"{s['correct'] - stats['baseline']['correct']:+.1f} pts vs plain RAG"
        col.metric(VARIANTS[v] + " · correct", f"{s['correct']:.1f}%", delta)

    st.dataframe(pd.DataFrame({
        VARIANTS[v]: {
            "Correct %": s["correct"], "Wrong %": s["incorrect"], "Refused %": s["refusal"],
            "Cites a gold evidence page %": s["cited_evidence_page"],
            "Used calculator %": s["used_calculator"], "LLM calls / question": s["llm_calls"],
            "Tokens / question": s["tokens"], "Seconds / question": s["seconds"],
        } for v, s in stats.items()}).round(1), width="stretch")
    st.caption("The agent's gain is statistically significant on numerical questions for both models and "
               "overall for the 4B model; see the README for the significance tests and results/RESULTS*.md "
               "for breakdowns by question type.")

    st.subheader("Question by question")
    by_id = {v: {r["id"]: r for r in rows} for v, rows in runs.items()}
    any_rows = by_id["agent"]
    table = pd.DataFrame([{
        "id": i.replace("financebench_id_", ""), "report": any_rows[i]["doc_name"],
        "type": any_rows[i]["question_type"], "question": any_rows[i]["question"],
        **{VARIANTS[v]: LABEL_ICON[by_id[v][i]["label"]] for v in by_id},
    } for i in sorted(common)])
    st.dataframe(table, width="stretch", hide_index=True)

    pick = st.selectbox("Inspect a question", sorted(common),
                        format_func=lambda i: f"{i.replace('financebench_id_', '')} · {any_rows[i]['question'][:100]}")
    q = any_rows[pick]
    st.markdown(f"**{q['question']}**")
    st.markdown(f"Reference answer: {q['gold']}  \nEvidence page(s): "
                f"{', '.join(str(p + 1) for p in q['gold_pages'])} of `{q['doc_name']}`")
    for v, col in zip(by_id, st.columns(len(by_id))):
        r = by_id[v][pick]
        with col:
            st.markdown(f"**{VARIANTS[v]}** · {LABEL_ICON[r['label']]}")
            with st.container(border=True):
                st.markdown(r["answer"][:2000])
            st.caption(f"Cited pages: {', '.join(str(p + 1) for p in r['pages']) or '-'} · "
                       f"{r['llm_calls']} LLM calls · {r['seconds']:.0f}s")
            if r.get("calcs"):
                st.caption("Calculator: " + "; ".join(r["calcs"]))
            with st.expander("Judge's reason"):
                st.write(r["judge_reason"])
            if r.get("trace"):
                with st.expander("Agent steps"):
                    for step in r["trace"]:
                        st.markdown("- " + describe(step))
