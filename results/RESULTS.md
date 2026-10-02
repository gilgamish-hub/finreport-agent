# FinanceBench results

Main model `groq:openai/gpt-oss-120b`, fast model `groq:openai/gpt-oss-20b`, judge `groq:qwen/qwen3.8-27b`. Each comparison uses only the questions both sides have graded. Graded so far: baseline 145, agent 145, agent_nocalc 30.
## Agent vs plain RAG (145 questions)

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 145 | 58.6 | 16.6 | 24.8 | 56.6 | 67.6 | 0.0 | 1.0 | 3,206 | 20.9 |
| agent | 145 | 63.4 | 25.5 | 11.0 | 64.1 | 72.4 | 54.5 | 5.0 | 10,508 | 47.4 |

By FinanceBench question type:

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline · domain-relevant | 47 | 57.4 | 17.0 | 25.5 | 46.8 | 51.1 | 0.0 | 1.0 | 3,014 | 20.6 |
| baseline · metrics-generated | 48 | 60.4 | 6.2 | 33.3 | 56.2 | 77.1 | 0.0 | 1.0 | 3,323 | 20.7 |
| baseline · novel-generated | 50 | 58.0 | 26.0 | 16.0 | 66.0 | 74.0 | 0.0 | 1.0 | 3,273 | 21.4 |
| agent · domain-relevant | 47 | 59.6 | 27.7 | 12.8 | 46.8 | 57.4 | 51.1 | 5.2 | 11,227 | 51.3 |
| agent · metrics-generated | 48 | 70.8 | 18.8 | 10.4 | 81.2 | 89.6 | 77.1 | 5.4 | 11,282 | 50.2 |
| agent · novel-generated | 50 | 60.0 | 30.0 | 10.0 | 64.0 | 70.0 | 36.0 | 4.3 | 9,089 | 41.1 |

Questions that need numerical reasoning:

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 64 | 50.0 | 10.9 | 39.1 | 42.2 | 56.2 | 0.0 | 1.0 | 3,192 | 20.7 |
| agent | 64 | 70.3 | 20.3 | 9.4 | 68.8 | 78.1 | 85.9 | 6.0 | 13,059 | 58.7 |

## Calculator ablation (30 questions)

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| agent_nocalc | 30 | 60.0 | 30.0 | 10.0 | 73.3 | 73.3 | 0.0 | 4.3 | 8,657 | 53.0 |
| agent | 30 | 63.3 | 23.3 | 13.3 | 66.7 | 73.3 | 56.7 | 4.8 | 9,883 | 41.4 |

By FinanceBench question type:

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| agent_nocalc · domain-relevant | 10 | 50.0 | 30.0 | 20.0 | 60.0 | 60.0 | 0.0 | 4.6 | 9,447 | 56.2 |
| agent_nocalc · metrics-generated | 10 | 60.0 | 30.0 | 10.0 | 80.0 | 80.0 | 0.0 | 4.2 | 8,376 | 48.8 |
| agent_nocalc · novel-generated | 10 | 70.0 | 30.0 | 0.0 | 80.0 | 80.0 | 0.0 | 4.2 | 8,147 | 54.0 |
| agent · domain-relevant | 10 | 60.0 | 20.0 | 20.0 | 50.0 | 60.0 | 40.0 | 4.4 | 8,809 | 42.4 |
| agent · metrics-generated | 10 | 60.0 | 20.0 | 20.0 | 70.0 | 80.0 | 80.0 | 5.8 | 11,582 | 48.3 |
| agent · novel-generated | 10 | 70.0 | 30.0 | 0.0 | 80.0 | 80.0 | 50.0 | 4.3 | 9,258 | 33.6 |

Questions that need numerical reasoning:

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| agent_nocalc | 16 | 62.5 | 25.0 | 12.5 | 75.0 | 75.0 | 0.0 | 4.4 | 9,222 | 53.7 |
| agent | 16 | 68.8 | 12.5 | 18.8 | 62.5 | 68.8 | 68.8 | 5.6 | 11,195 | 49.6 |

Right with the calculator but wrong without it: **2** (financebench_id_00438, financebench_id_06655). Right without it but wrong with it: **1** (financebench_id_04481).
