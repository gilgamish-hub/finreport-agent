"""Run the agent and/or plain RAG on FinanceBench questions and grade the answers.

    python scripts/run_eval.py --variants agent baseline --sample 30
    python scripts/run_eval.py --variants agent_nocalc --sample 30    # ablation: no calculator tool

Results go to results/<variant>.jsonl, one line per question. Re-running skips questions that are
already in the file, so a run interrupted by free-tier rate limits can simply be started again.
"""

import argparse
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent import config  # noqa: E402
from finagent.baseline import run_baseline  # noqa: E402
from finagent.data import load_questions  # noqa: E402
from finagent.evaluate import judge, load_results  # noqa: E402
from finagent.graph import build_graph, run_agent  # noqa: E402
from finagent.ingest import indexed_docs  # noqa: E402
from finagent.llm import CallCounter, get_llm  # noqa: E402

VARIANTS = ["agent", "agent_nocalc", "baseline"]

# Questions inspected while debugging and tuning prompts. They are kept out of --sample so the
# reported numbers come from questions the prompts were not tuned on.
DEV_IDS = {"financebench_id_03029", "financebench_id_00799", "financebench_id_01079",
           "financebench_id_01226", "financebench_id_02987"}


def sample_questions(questions, n, seed=0):
    """Stratified by question_type, so a small sample still mixes extraction, reasoning and domain questions."""
    groups = defaultdict(list)
    for q in sorted(questions, key=lambda q: q.id):
        groups[q.question_type].append(q)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out, i = [], 0
    while len(out) < n and any(i < len(g) for g in groups.values()):
        for g in groups.values():
            if i < len(g) and len(out) < n:
                out.append(g[i])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["agent", "baseline"], choices=VARIANTS)
    ap.add_argument("--sample", type=int, help="stratified sample of N questions (default: all 150)")
    ap.add_argument("--ids", nargs="*", help="specific financebench_ids")
    ap.add_argument("--tag", default="", help="suffix for the results files, e.g. a model name")
    args = ap.parse_args()

    questions = load_questions()
    if args.ids:
        questions = [q for q in questions if q.id in set(args.ids)]
    elif args.sample:
        questions = sample_questions([q for q in questions if q.id not in DEV_IDS], args.sample)
    available = indexed_docs()
    skipped = [q for q in questions if q.doc_name not in available]
    questions = [q for q in questions if q.doc_name in available]
    if skipped:
        print(f"Skipping {len(skipped)} questions whose report is not indexed yet.")

    main_llm, fast_llm = get_llm(config.MAIN_LLM), get_llm(config.FAST_LLM)
    # max_tokens keeps Groq's per-request output estimate under the free-tier output-tokens-per-minute cap
    judge_llm = get_llm(config.JUDGE_LLM, max_tokens=1000)
    graphs = {
        "agent": build_graph(main_llm, fast_llm, use_calculator=True),
        "agent_nocalc": build_graph(main_llm, fast_llm, use_calculator=False),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)

    for variant in args.variants:
        out_path = config.RESULTS_DIR / f"{variant}{args.tag}.jsonl"
        done = {r["id"] for r in load_results(out_path) if "label" in r}
        todo = [q for q in questions if q.id not in done]
        print(f"\n== {variant}: {len(done)} done, {len(todo)} to run -> {out_path.name}")

        for i, q in enumerate(todo, 1):
            counter = CallCounter()
            t = time.time()
            try:
                if variant == "baseline":
                    res = run_baseline(main_llm, q.question, q.doc_name, callbacks=[counter])
                else:
                    res = run_agent(graphs[variant], q.question, q.doc_name, callbacks=[counter])
                seconds = time.time() - t
                verdict = judge(judge_llm, q.question, q.answer, q.justification, res["answer"])
            except Exception as e:
                print(f"  [{i}/{len(todo)}] {q.id} ERROR {type(e).__name__}: {str(e)[:200]}")
                msg = str(e).lower()
                if "quota" in msg or "per day" in msg:
                    print("  Daily limit reached. Re-run later (tomorrow) to continue.")
                    return
                traceback.print_exc(limit=2)
                continue

            row = {
                "id": q.id, "doc_name": q.doc_name, "question": q.question, "gold": q.answer,
                "question_type": q.question_type, "reasoning": q.reasoning, "gold_pages": q.evidence_pages,
                "answer": res["answer"], "status": res["status"], "pages": res["pages"],
                "retrieved_pages": res["retrieved_pages"], "calcs": res["calcs"], "qtype": res.get("qtype"),
                "trace": res["trace"], "llm_calls": counter.calls, "tokens": counter.tokens, "seconds": round(seconds, 1),
                "label": verdict.label, "judge_reason": verdict.reason,
                "model": config.MAIN_LLM, "fast_model": config.FAST_LLM, "judge_model": config.JUDGE_LLM,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(todo)}] {q.id} {verdict.label:9s} calls={counter.calls} tokens={counter.tokens} {seconds:.0f}s")


if __name__ == "__main__":
    main()
