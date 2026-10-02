"""Turn results/*.jsonl into results/RESULTS.md.

Only questions answered by every variant are compared, so the numbers are like for like.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent import config  # noqa: E402
from finagent.evaluate import by_group, calculator_fixes, load_results, summarize  # noqa: E402


def table(summaries: dict[str, dict]) -> str:
    cols = [("n", "n", "{:.0f}"), ("correct", "Correct %", "{:.1f}"), ("incorrect", "Wrong %", "{:.1f}"),
            ("refusal", "Refused %", "{:.1f}"), ("cited_evidence_page", "Cites gold page %", "{:.1f}"),
            ("retrieved_evidence_page", "Retrieved gold page %", "{:.1f}"),
            ("used_calculator", "Used calculator %", "{:.1f}"), ("llm_calls", "LLM calls / q", "{:.1f}"), ("tokens", "Tokens / q", "{:,.0f}"),
            ("seconds", "Seconds / q", "{:.1f}")]
    lines = ["| | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, s in summaries.items():
        if not s.get("n"):
            continue
        lines.append(f"| {name} | " + " | ".join(c[2].format(s[c[0]]) for c in cols) + " |")
    return "\n".join(lines)


def paired(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Restrict two runs to the questions both have graded, so they are compared like for like."""
    common = {r["id"] for r in a} & {r["id"] for r in b}
    return [r for r in a if r["id"] in common], [r for r in b if r["id"] in common]


def numerical(rows: list[dict]) -> list[dict]:
    return [r for r in rows if "numerical" in (r.get("reasoning") or "").lower()]


def comparison(name_a: str, a: list[dict], name_b: str, b: list[dict]) -> list[str]:
    out = ["", table({name_a: summarize(a), name_b: summarize(b)}), "",
           "By FinanceBench question type:", "",
           table({f"{n} · {t}": s for n, rows in [(name_a, a), (name_b, b)]
                  for t, s in by_group(rows, "question_type").items()}), ""]
    if numerical(a):
        out += ["Questions that need numerical reasoning:", "",
                table({name_a: summarize(numerical(a)), name_b: summarize(numerical(b))}), ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    runs = {v: load_results(config.RESULTS_DIR / f"{v}{args.tag}.jsonl") for v in ["baseline", "agent", "agent_nocalc"]}
    runs = {k: [r for r in v if "label" in r] for k, v in runs.items() if v}
    if "agent" not in runs:
        sys.exit("No agent results yet. Run scripts/run_eval.py first.")

    model = runs["agent"][0]
    out = [f"# FinanceBench results{' (' + args.tag.strip('_') + ')' if args.tag else ''}", "",
           f"Main model `{model['model']}`, fast model `{model['fast_model']}`, "
           f"judge `{model.get('judge_model', '-')}`. Each comparison uses only the questions both sides "
           "have graded. Graded so far: " + ", ".join(f"{k} {len(v)}" for k, v in runs.items()) + "."]

    if "baseline" in runs:
        agent, base = paired(runs["agent"], runs["baseline"])
        out += [f"## Agent vs plain RAG ({len(agent)} questions)"] + comparison("baseline", base, "agent", agent)

    if "agent_nocalc" in runs:
        agent, nocalc = paired(runs["agent"], runs["agent_nocalc"])
        fx = calculator_fixes(agent, nocalc)
        out += [f"## Calculator ablation ({len(agent)} questions)"] + comparison("agent_nocalc", nocalc, "agent", agent)
        out += [f"Right with the calculator but wrong without it: **{len(fx['fixed'])}** "
                f"({', '.join(fx['fixed']) or '-'}). Right without it but wrong with it: "
                f"**{len(fx['broke'])}** ({', '.join(fx['broke']) or '-'}).", ""]

    path = config.RESULTS_DIR / f"RESULTS{args.tag}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
