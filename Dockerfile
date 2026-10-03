# FinReport Agent API (finagent/api.py) as a container.
#
#   docker build -t finreport-api .
#   docker run -p 8000:8000 --env-file .env finreport-api        then open http://localhost:8000/docs
#
# On first start the 12-filing demo index (78 MB) is downloaded from the Hugging Face Hub.
# To use your full local index instead:  -v "$(pwd)/data/chroma:/app/data/demo_chroma"

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# Run as a normal user, not root
RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app
WORKDIR /app
USER app
ENV PATH=/home/app/.local/bin:$PATH

# Dependencies first: this layer is cached and only rebuilt when requirements change.
# requirements.txt points pip at the CPU-only PyTorch build (no 2 GB of CUDA libraries).
COPY --chown=app requirements.txt .
RUN pip install --user -r requirements.txt

# Bake the embedding model into the image so the container starts without downloading it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY --chown=app finagent/ finagent/

ENV FINAGENT_CHROMA_DIR=/app/data/demo_chroma \
    FINAGENT_INDEX_REPO=GILGAMISH/finreport-demo-index \
    FINAGENT_LLM=groq:openai/gpt-oss-120b \
    FINAGENT_FAST_LLM=groq:openai/gpt-oss-20b

EXPOSE 8000
# Cloud hosts pass the port in $PORT; locally it defaults to 8000
CMD uvicorn finagent.api:app --host 0.0.0.0 --port ${PORT:-8000}
