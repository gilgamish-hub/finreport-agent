"""What routing would have scored, computed from the saved evaluation runs (no API calls).

Router rule: the agent's own classify step labels each question lookup / calculate / compare.
Lookups go to plain RAG, everything else through the full agent loop. Because both paths were run on
every question, the router's answer to each question is already known: the baseline's answer for lookups,
the agent's answer otherwise.

    python scripts/simulate_router.py            # gpt-oss-120b runs
    python scripts/simulate_router.py --tag _ollama

Caveat: the rule was suggested by these same results (the agent helped on calculations, not on lookups),
so this is an estimate. `run_eval.py --variants router` runs the router for real.
"""

import argparse
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent import config  # noqa: E402
from finagent.evaluate import load_results  # noqa: E402

# The routed lookups still pay for the classify call that decides the route.
# Measured on 10 lookup questions with gpt-oss-20b: 969 tokens on average.
CLASSIFY_TOKENS = 970


def mcnemar(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    agent = {r["id"]: r for r in load_results(config.RESULTS_DIR / f"agent{args.tag}.jsonl") if "label" in r}
    base = {r["id"]: r for r in load_results(config.RESULTS_DIR / f"baseline{args.tag}.jsonl") if "label" in r}
    ids = sorted(agent.keys() & base.keys())

    router = {}
    for i in ids:
        if agent[i]["qtype"] == "lookup":
            r = dict(base[i])
            r["tokens"] = r.get("tokens", 0) + CLASSIFY_TOKENS
            r["llm_calls"] = r.get("llm_calls", 0) + 1
            router[i] = r
        else:
            router[i] = agent[i]

    n = len(ids)
    pct = lambda d, label: 100 * sum(d[i]["label"] == label for i in ids) / n  # noqa: E731
    avg = lambda d, key: sum(d[i].get(key, 0) for i in ids) / n  # noqa: E731
    lookups = sum(agent[i]["qtype"] == "lookup" for i in ids)

    out = [f"# Routing estimate ({config.RESULTS_DIR.name}/agent{args.tag}.jsonl, {n} questions)", "",
           f"{lookups} of {n} questions were classified as lookups and routed to plain RAG.", "",
           "| | Correct % | Wrong % | Refused % | LLM calls / q | Tokens / q |", "|---|---|---|---|---|---|"]
    for name, d in [("Plain RAG", base), ("Agent", agent), ("Router", router)]:
        out.append(f"| {name} | {pct(d, 'correct'):.1f} | {pct(d, 'incorrect'):.1f} | {pct(d, 'refusal'):.1f} "
                   f"| {avg(d, 'llm_calls'):.1f} | {avg(d, 'tokens'):,.0f} |")
    out.append("")
    for name, other in [("plain RAG", base), ("agent", agent)]:
        b = sum(router[i]["label"] == "correct" and other[i]["label"] != "correct" for i in ids)
        c = sum(router[i]["label"] != "correct" and other[i]["label"] == "correct" for i in ids)
        out.append(f"- Router vs {name}: {b} questions won, {c} lost, exact McNemar p = {mcnemar(b, c):.3f}")
    out += ["", "Estimate from saved runs; the routing rule was suggested by these same results."]

    text = "\n".join(out) + "\n"
    (config.RESULTS_DIR / f"ROUTER{args.tag}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
