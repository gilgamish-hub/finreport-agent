"""Download the FinanceBench annual-report PDFs.

    python scripts/download_pdfs.py            # every report that has a question
    python scripts/download_pdfs.py --docs 3M_2018_10K AMCOR_2023_10K
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent.data import download_pdf, load_questions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", help="doc_names to fetch (default: all)")
    args = ap.parse_args()

    docs = args.docs or sorted({q.doc_name for q in load_questions()})
    print(f"Downloading {len(docs)} reports...")
    failed = []

    def fetch(doc):
        try:
            p = download_pdf(doc)
            return doc, p.stat().st_size, None
        except Exception as e:
            return doc, 0, e

    with ThreadPoolExecutor(max_workers=4) as pool:
        for doc, size, err in pool.map(fetch, docs):
            if err:
                failed.append(doc)
                print(f"  FAILED {doc}: {err}")
            else:
                print(f"  {doc}  {size / 1e6:.1f} MB")
    print(f"Done. {len(docs) - len(failed)} ok, {len(failed)} failed.")


if __name__ == "__main__":
    main()
