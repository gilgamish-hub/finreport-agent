# Routing estimate (results/agent.jsonl, 145 questions)

56 of 145 questions were classified as lookups and routed to plain RAG.

| | Correct % | Wrong % | Refused % | LLM calls / q | Tokens / q |
|---|---|---|---|---|---|
| Plain RAG | 58.6 | 16.6 | 24.8 | 1.0 | 3,206 |
| Agent | 63.4 | 25.5 | 11.0 | 5.0 | 10,508 |
| Router | 67.6 | 21.4 | 11.0 | 4.2 | 8,975 |

- Router vs plain RAG: 20 questions won, 7 lost, exact McNemar p = 0.019
- Router vs agent: 10 questions won, 4 lost, exact McNemar p = 0.180

Estimate from saved runs; the routing rule was suggested by these same results.
