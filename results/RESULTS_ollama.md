# FinanceBench results (ollama)

Main model `ollama:qwen3.5:4b`, fast model `ollama:qwen3.5:4b`, judge `groq:qwen/qwen3.8-27b`. Each comparison uses only the questions both sides have graded. Graded so far: baseline 145, agent 145, agent_nocalc 145.
## Agent vs plain RAG (145 questions)

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 145 | 46.2 | 24.8 | 29.0 | 61.4 | 67.6 | 0.0 | 1.0 | 3,615 | 13.0 |
| agent | 145 | 62.1 | 24.1 | 13.8 | 62.8 | 75.9 | 95.9 | 5.0 | 14,041 | 29.2 |

By FinanceBench question type:

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline · domain-relevant | 47 | 40.4 | 27.7 | 31.9 | 48.9 | 51.1 | 0.0 | 1.0 | 3,117 | 8.1 |
| baseline · metrics-generated | 48 | 50.0 | 12.5 | 37.5 | 66.7 | 77.1 | 0.0 | 1.0 | 4,010 | 18.4 |
| baseline · novel-generated | 50 | 48.0 | 34.0 | 18.0 | 68.0 | 74.0 | 0.0 | 1.0 | 3,704 | 12.6 |
| agent · domain-relevant | 47 | 59.6 | 19.1 | 21.3 | 38.3 | 57.4 | 100.0 | 5.2 | 14,752 | 34.1 |
| agent · metrics-generated | 48 | 72.9 | 18.8 | 8.3 | 83.3 | 93.8 | 89.6 | 5.0 | 14,632 | 26.7 |
| agent · novel-generated | 50 | 54.0 | 34.0 | 12.0 | 66.0 | 76.0 | 98.0 | 4.8 | 12,804 | 27.0 |

Questions that need numerical reasoning:

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 64 | 35.9 | 15.6 | 48.4 | 48.4 | 56.2 | 0.0 | 1.0 | 3,665 | 17.2 |
| agent | 64 | 65.6 | 21.9 | 12.5 | 62.5 | 79.7 | 95.3 | 5.2 | 15,948 | 35.6 |

## Calculator ablation (145 questions)

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| agent_nocalc | 145 | 57.9 | 26.9 | 15.2 | 62.1 | 75.9 | 0.0 | 4.0 | 10,910 | 36.9 |
| agent | 145 | 62.1 | 24.1 | 13.8 | 62.8 | 75.9 | 95.9 | 5.0 | 14,041 | 29.2 |

By FinanceBench question type:

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| agent_nocalc · domain-relevant | 47 | 57.4 | 25.5 | 17.0 | 48.9 | 57.4 | 0.0 | 4.0 | 10,054 | 20.5 |
| agent_nocalc · metrics-generated | 48 | 64.6 | 18.8 | 16.7 | 77.1 | 93.8 | 0.0 | 4.0 | 12,350 | 52.2 |
| agent_nocalc · novel-generated | 50 | 52.0 | 36.0 | 12.0 | 60.0 | 76.0 | 0.0 | 3.9 | 10,332 | 37.5 |
| agent · domain-relevant | 47 | 59.6 | 19.1 | 21.3 | 38.3 | 57.4 | 100.0 | 5.2 | 14,752 | 34.1 |
| agent · metrics-generated | 48 | 72.9 | 18.8 | 8.3 | 83.3 | 93.8 | 89.6 | 5.0 | 14,632 | 26.7 |
| agent · novel-generated | 50 | 54.0 | 34.0 | 12.0 | 66.0 | 76.0 | 98.0 | 4.8 | 12,804 | 27.0 |

Questions that need numerical reasoning:

| | n | Correct % | Wrong % | Refused % | Cites gold page % | Retrieved gold page % | Used calculator % | LLM calls / q | Tokens / q | Seconds / q |
|---|---|---|---|---|---|---|---|---|---|---|
| agent_nocalc | 64 | 59.4 | 21.9 | 18.8 | 64.1 | 79.7 | 0.0 | 4.1 | 12,226 | 44.4 |
| agent | 64 | 65.6 | 21.9 | 12.5 | 62.5 | 79.7 | 95.3 | 5.2 | 15,948 | 35.6 |

Right with the calculator but wrong without it: **21** (financebench_id_00070, financebench_id_00222, financebench_id_00382, financebench_id_00651, financebench_id_00669, financebench_id_00724, financebench_id_00790, financebench_id_00822, financebench_id_01107, financebench_id_01244, financebench_id_01351, financebench_id_02024, financebench_id_02419, financebench_id_03069, financebench_id_03838, financebench_id_04171, financebench_id_04481, financebench_id_06655, financebench_id_07507, financebench_id_10130, financebench_id_10136). Right without it but wrong with it: **15** (financebench_id_00215, financebench_id_00521, financebench_id_00591, financebench_id_00601, financebench_id_00685, financebench_id_00702, financebench_id_00723, financebench_id_00757, financebench_id_00882, financebench_id_00917, financebench_id_00956, financebench_id_02608, financebench_id_03718, financebench_id_04735, financebench_id_04980).
