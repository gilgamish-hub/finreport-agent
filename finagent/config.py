"""Central settings. Everything can be overridden with environment variables (or a .env file)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("FINAGENT_DATA_DIR", ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = Path(os.getenv("FINAGENT_CHROMA_DIR", DATA_DIR / "chroma"))
RESULTS_DIR = ROOT / "results"

COLLECTION = "financebench"
PDF_URL = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc_name}.pdf"
HF_DATASET = "PatronusAI/financebench"

# Local embedding model (runs on CPU, no API cost).
EMBED_MODEL = os.getenv("FINAGENT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

# LLMs, given as "<provider>:<model>" for langchain's init_chat_model.
# The "fast" model plans, grades, rewrites and verifies; the "main" model reasons and answers;
# the judge grades answers in the evaluation (best from a different model family than the main one).
# The app defaults to Gemini; the evaluation used Groq's free tier (see .env.example).
# Offline alternative: FINAGENT_LLM=ollama:qwen3.5:4b
MAIN_LLM = os.getenv("FINAGENT_LLM", "google_genai:gemini-3.5-flash")
FAST_LLM = os.getenv("FINAGENT_FAST_LLM", os.getenv("FINAGENT_LLM", "google_genai:gemini-3.5-flash-lite"))
JUDGE_LLM = os.getenv("FINAGENT_JUDGE_LLM", MAIN_LLM)
# Requests per minute allowed per model. Free tiers: Gemini is the tightest, Groq allows ~30.
DEFAULT_RPM = {"google_genai": 8, "groq": 25}
LLM_RPM = os.getenv("FINAGENT_RPM")
# Ollama only: force this many layers onto the GPU (99 = all). Unset = let Ollama decide.
OLLAMA_NUM_GPU = os.getenv("FINAGENT_OLLAMA_NUM_GPU")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 5                  # chunks per search query
MAX_CONTEXT_CHUNKS = 8     # new chunks graded per search round
MAX_KEPT_CHUNKS = 10
# Full-page evidence for the reasoning step is capped (~4k tokens) to fit free-tier token-per-minute limits.
MAX_EVIDENCE_CHARS = int(os.getenv("FINAGENT_MAX_EVIDENCE_CHARS", "16000"))
MAX_RETRIEVAL_ROUNDS = 3   # first search + up to 2 rewrites
MAX_REASON_STEPS = 6       # LLM turns inside the reason/calculator loop
MAX_VERIFY_RETRIES = 1

MAX_UPLOAD_PAGES = int(os.getenv("FINAGENT_MAX_UPLOAD_PAGES", "150"))
UPLOAD_DIR = DATA_DIR / "uploads"

NOT_FOUND = "I couldn't find this in the report."
