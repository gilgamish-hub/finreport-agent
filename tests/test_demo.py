import json

import pytest

from finagent.demo import (AllModelsBusy, AnswerCache, ModelConfig, is_rate_limit, load_saved_runs, normalize,
                           run_with_fallback)

A = ModelConfig("A", "groq:a", "groq:a", "GROQ_API_KEY")
B = ModelConfig("B", "groq:b", "groq:b", "GROQ_API_KEY")


class RateLimitError(Exception):
    pass


def test_is_rate_limit_recognises_provider_errors():
    assert is_rate_limit(RateLimitError("Error code: 429 - tokens per day (TPD)"))
    assert is_rate_limit(Exception("429 RESOURCE_EXHAUSTED. You exceeded your current quota"))
    assert is_rate_limit(Exception("Error code: 413 - Request too large for model"))
    assert not is_rate_limit(ValueError("doc is not indexed"))


def test_fallback_moves_to_next_model_on_rate_limit():
    switches = []

    def run(cfg):
        if cfg is A:
            raise RateLimitError("429 tokens per day")
        return "answer from " + cfg.label

    result, used = run_with_fallback([A, B], run, on_switch=lambda a, b: switches.append((a.label, b.label)))
    assert (result, used) == ("answer from B", B)
    assert switches == [("A", "B")]


def test_fallback_raises_when_every_model_is_busy():
    def run(cfg):
        raise RateLimitError("429")

    with pytest.raises(AllModelsBusy):
        run_with_fallback([A, B], run)


def test_fallback_does_not_hide_real_bugs():
    def run(cfg):
        raise KeyError("bug")

    with pytest.raises(KeyError):
        run_with_fallback([A, B], run)


def test_cache_round_trip_ignores_case_and_spacing(tmp_path):
    cache = AnswerCache(tmp_path / "c.sqlite")
    cache.put("3M_2018_10K", "What was capex in FY2018?", {"answer": "$1,577 million"})
    assert cache.get("3M_2018_10K", "  what was CAPEX in   FY2018 ") == {"answer": "$1,577 million"}
    assert cache.get("OTHER_DOC", "What was capex in FY2018?") is None


def test_saved_runs_skip_failed_rows(tmp_path):
    rows = [{"id": "1", "doc_name": "D", "question": "Q one?", "label": "correct", "answer": "a"},
            {"id": "2", "doc_name": "D", "question": "Q two?", "label": "incorrect", "error": True, "answer": "x"}]
    (tmp_path / "agent.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    saved = load_saved_runs(tmp_path)
    assert set(saved) == {("D", normalize("Q one?"))}
