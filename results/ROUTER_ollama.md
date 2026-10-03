# Routing estimate (results/agent_ollama.jsonl, 145 questions)

43 of 145 questions were classified as lookups and routed to plain RAG.

| | Correct % | Wrong % | Refused % | LLM calls / q | Tokens / q |
|---|---|---|---|---|---|
| Plain RAG | 46.2 | 24.8 | 29.0 | 1.0 | 3,615 |
| Agent | 62.1 | 24.1 | 13.8 | 5.0 | 14,041 |
| Router | 60.0 | 28.3 | 11.7 | 4.2 | 12,097 |

- Router vs plain RAG: 27 questions won, 7 lost, exact McNemar p = 0.001
- Router vs agent: 3 questions won, 6 lost, exact McNemar p = 0.508

Estimate from saved runs; the routing rule was suggested by these same results.
