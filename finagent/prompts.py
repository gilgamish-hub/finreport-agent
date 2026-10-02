CLASSIFY = """You plan searches over ONE company filing: {doc_name}.

Question: {question}

1. Classify the question:
   - lookup:    one fact or figure that is stated directly in the report
   - calculate: needs arithmetic on one or more figures (ratio, margin, growth, average, sum)
   - compare:   compares figures across years, segments or items
2. Write 1-4 search queries, one per distinct figure or fact you need. Phrase them the way a 10-K/10-Q
   would (financial-statement line items, note titles, segment names) and include the fiscal year.
   Example for "FY2018 capex": "purchases of property, plant and equipment 2018 cash flow statement".
   For "why" / "what drove" / qualitative questions, also search for the narrative explanation
   (e.g. "results of operations operating margin change 2022 MD&A"), not only the figures."""

GRADE = """You check whether retrieved passages from {doc_name} are enough to answer a question.

Question: {question}
{kept}
New passages:
{candidates}

Return:
- relevant: indices of NEW passages that contain information needed for the answer (figures, line
  items, statements). Ignore passages that only mention the topic.
- sufficient: true only if the kept + relevant passages together contain EVERY figure or fact needed.
- missing: if not sufficient, what exactly is still missing (e.g. "total revenue for FY2022")."""

REWRITE = """Searches over {doc_name} have not found everything needed.

Question: {question}
Queries already tried: {tried}
Still missing: {missing}

Write 1-3 NEW search queries for the missing information. Use different wording from the tried
queries: the exact line-item names a 10-K uses (e.g. "net sales" instead of "revenue", "purchases of
property, plant and equipment" instead of "capex"), the statement or note it would sit in, and the year.
Typical places: income statement, balance sheet, cash flow statement, "Results of Operations" and
"Liquidity and Capital Resources" (MD&A), notes such as "Acquisitions and Divestitures", "Segment
Information", "Debt", "Restructuring", "Income Taxes". Name sections by title; do not guess item numbers."""

REASON_SYSTEM = """You are a careful financial analyst answering questions about one company filing
({doc_name}) using ONLY the evidence passages provided. Each passage starts with [doc | page N].

Rules:
- Take every figure from the evidence. Never use outside knowledge about the company.
{calc_rule}
- State units (USD millions, %, etc.) and round sensibly (two decimals for ratios/percentages unless asked).
- If the evidence answers only part of the question, answer that part and say clearly what the
  report excerpts do not show. Only if the evidence contains nothing useful, call submit_answer with
  answer exactly "{not_found}" and pages [].
- Finish by calling submit_answer with the answer and the page numbers you used."""

CALC_RULE_TOOL = """- Do ALL arithmetic with the calculator tool, one call per calculation; when several calculations
  are independent, make all those calls in the same turn. Never compute in your head."""
CALC_RULE_NO_TOOL = "- Work out any arithmetic yourself, step by step."

REASON_USER = """Evidence:
{evidence}

Question: {question}"""

VERIFY = """Check an answer to a question about {doc_name} against the evidence.

Question: {question}
Answer: {answer}
Calculator results used: {calcs}

Evidence:
{evidence}

Is every number and claim in the answer either stated in the evidence or correctly computed from
figures in the evidence? Small rounding differences and unit conversions are fine.
Return supported=true/false and, if false, the specific problem."""

BASELINE = """Answer the question about the company filing {doc_name} using only the passages below.
Each passage starts with [doc | page N]. If the passages do not contain the answer, reply exactly
"{not_found}".

{evidence}

Question: {question}

Give a concise answer with units. On the last line write "Pages:" followed by the page numbers you used."""

JUDGE = """You grade answers to questions about company financial filings.

Question: {question}
Reference answer: {gold}
Reference justification: {justification}

Model answer: {pred}

Label the model answer:
- correct:   it gives the same answer as the reference. Numbers may differ only by rounding or by an
             equivalent unit (1,577 million = 1.58 billion). For yes/no or qualitative questions the
             conclusion and key supporting facts must match; extra detail is fine.
- incorrect: it gives a different, wrong or contradictory answer.
- refusal:   it declines or says the information could not be found."""
