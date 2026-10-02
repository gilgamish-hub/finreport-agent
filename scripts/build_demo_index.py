"""Copy a few reports from the full index into a small one for the hosted demo.

The full index (84 filings, ~440 MB) is too big for free hosting. This copies the stored chunks and
embeddings of the filings with the most evaluated questions, so nothing is re-embedded.

    python scripts/build_demo_index.py               # top 12 filings
    python scripts/build_demo_index.py --top 8
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from finagent import config  # noqa: E402
from finagent.demo import load_saved_runs  # noqa: E402

DEMO_DIR = config.DATA_DIR / "demo_chroma"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", type=Path, default=DEMO_DIR)
    args = ap.parse_args()

    counts = Counter(doc for doc, _ in load_saved_runs())
    docs = [d for d, _ in counts.most_common(args.top)]
    print("Filings:", ", ".join(f"{d} ({counts[d]} q)" for d in docs))

    src = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.COLLECTION)
    if args.out.exists():
        shutil.rmtree(args.out)
    dst = chromadb.PersistentClient(path=str(args.out)).create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"})

    total = 0
    for doc in docs:
        got = src.get(where={"doc_name": doc}, include=["embeddings", "documents", "metadatas"])
        dst.add(ids=got["ids"], embeddings=got["embeddings"], documents=got["documents"],
                metadatas=got["metadatas"])
        total += len(got["ids"])
        print(f"  {doc}: {len(got['ids'])} chunks")

    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"Done: {total} chunks, {size:.0f} MB in {args.out}")
    print(f"Use it with FINAGENT_CHROMA_DIR={args.out}")


if __name__ == "__main__":
    main()
