"""Embed report PDFs into the local Chroma store.

    python scripts/build_index.py                 # every downloaded FinanceBench report
    python scripts/build_index.py --docs 3M_2018_10K
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent.data import download_pdf, load_questions  # noqa: E402
from finagent.ingest import index_pdf, indexed_docs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", help="doc_names to index (default: all in FinanceBench)")
    args = ap.parse_args()

    companies = {q.doc_name: q.company for q in load_questions()}
    docs = args.docs or sorted(companies)
    print(f"{len(indexed_docs() & set(docs))} of {len(docs)} reports already in the index (complete ones are skipped).")

    for i, doc in enumerate(docs, 1):
        t = time.time()
        n = index_pdf(download_pdf(doc), doc, companies.get(doc, ""))
        status = f"{n} chunks in {time.time() - t:.0f}s" if n else "already indexed"
        print(f"[{i}/{len(docs)}] {doc}: {status}", flush=True)


if __name__ == "__main__":
    main()
