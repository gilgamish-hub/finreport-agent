"""Chat-model factory. Any provider langchain's init_chat_model knows works
("google_genai:...", "groq:...", "ollama:...")."""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter

from finagent import config

_limiters: dict[str, InMemoryRateLimiter] = {}


def _limiter(model: str, provider: str) -> InMemoryRateLimiter:
    # One limiter per model, since free-tier quotas are counted per model.
    if model not in _limiters:
        rpm = float(config.LLM_RPM or config.DEFAULT_RPM.get(provider, 30))
        _limiters[model] = InMemoryRateLimiter(
            requests_per_second=rpm / 60, check_every_n_seconds=0.2, max_bucket_size=2
        )
    return _limiters[model]


def get_llm(model: str | None = None, api_key: str | None = None, temperature: float = 0.0,
            max_tokens: int | None = None):
    model = model or config.MAIN_LLM
    provider, name = model.split(":", 1)
    kwargs: dict = {"temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if provider == "google_genai":
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY (free key: https://aistudio.google.com/apikey).")
        kwargs.update(google_api_key=key, max_retries=6, rate_limiter=_limiter(model, provider))
    elif provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY (free key: https://console.groq.com/keys).")
        # Free tier counts tokens per minute; the client waits and retries on 429s.
        kwargs.update(api_key=key, max_retries=10, rate_limiter=_limiter(model, provider))
        if name.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "medium"
    elif provider == "ollama":
        kwargs.update(reasoning=False, num_ctx=16384)
        if config.OLLAMA_NUM_GPU:   # layers on the GPU; Ollama's own estimate can leave a 4 GB card half used
            kwargs["num_gpu"] = int(config.OLLAMA_NUM_GPU)
    return init_chat_model(model, **kwargs)


def structured(llm, schema):
    """`llm.with_structured_output(schema)` that survives the odd malformed reply.

    On Groq, native JSON-schema mode is far more reliable than the default forced tool call
    (gpt-oss-20b sometimes answers in prose and the request fails).
    """
    kwargs = {"method": "json_schema"} if type(llm).__name__ == "ChatGroq" else {}
    return llm.with_structured_output(schema, **kwargs).with_retry(stop_after_attempt=3)


class CallCounter(BaseCallbackHandler):
    """Counts chat-model calls and tokens, so the evaluation can report the cost of each approach."""

    def __init__(self):
        self.calls = 0
        self.tokens = 0

    def on_chat_model_start(self, *args, **kwargs):
        self.calls += 1

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None)
                if usage:
                    self.tokens += usage.get("total_tokens", 0)
