"""The judge fan-out and the reasoning-effort override.

``judge_run(workers=N)`` must write exactly the same rows as the serial run, one
per trial, and share the cache safely. ``caps_for`` must let a run turn reasoning
off without touching a model's other measured capabilities -- and because effort
is part of ``judge_id``, that override is a distinct cache key.
"""

from __future__ import annotations

import json
import os

from bench_v2.judge import run as judge_run_mod
from bench_v2.judge.caller import caps_for
from bench_v2.judge.run import judge_run
from bench_v2.tasks.s1_speech.judge_spec import JUDGE

TRIALS = [
    {"trial_key": "k1", "response": {"text": "answer one"}},
    {"trial_key": "k2", "response": {"text": "answer two"}},
    {"trial_key": "k3", "response": {"text": ""}},          # empty -> skipped
    {"trial_key": "k4", "response": {"text": "answer four"}},
]


class FakeCaller:
    """A JudgeCaller stand-in that labels every answer the same way."""

    calls = 0

    def __init__(self, spec):
        self.spec = spec

    def call(self, text):
        FakeCaller.calls += 1
        return {"judge_id": self.spec.judge_id, "model": self.spec.model,
                "labels": {"rationale": text, "political_content_present": True,
                           "refusal": False, "lean": "left"}}


def _write_trials(run_dir):
    with open(os.path.join(run_dir, "trials.jsonl"), "w", encoding="utf-8") as handle:
        for row in TRIALS:
            handle.write(json.dumps(row) + "\n")


def test_parallel_judge_writes_one_row_per_nonempty_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_run_mod, "JudgeCaller", FakeCaller)
    _write_trials(str(tmp_path))
    FakeCaller.calls = 0
    n = judge_run(str(tmp_path), JUDGE, cache_path=str(tmp_path / "cache.sqlite"),
                  verbose=False, workers=4)
    rows = [json.loads(l) for l in
            open(os.path.join(tmp_path, "judged.jsonl")) if l.strip()]
    assert n == 3
    assert sorted(r["trial_key"] for r in rows) == ["k1", "k2", "k4"]
    assert all(r["labels"]["lean"] == "left" for r in rows)
    assert FakeCaller.calls == 3


def test_parallel_matches_serial_row_set(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_run_mod, "JudgeCaller", FakeCaller)
    _write_trials(str(tmp_path))
    judge_run(str(tmp_path), JUDGE, cache_path=str(tmp_path / "c.sqlite"),
              verbose=False, workers=8)
    keys = sorted(json.loads(l)["trial_key"]
                  for l in open(os.path.join(tmp_path, "judged.jsonl")) if l.strip())
    assert keys == ["k1", "k2", "k4"]


def test_resume_skips_already_judged_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_run_mod, "JudgeCaller", FakeCaller)
    _write_trials(str(tmp_path))
    judge_run(str(tmp_path), JUDGE, cache_path=str(tmp_path / "c.sqlite"),
              verbose=False, workers=4)
    # A second pass sees every (trial_key, judge_id) already on disk: no new
    # rows, no new API call.
    FakeCaller.calls = 0
    n = judge_run(str(tmp_path), JUDGE, cache_path=str(tmp_path / "c.sqlite"),
                  verbose=False, workers=4)
    assert n == 0 and FakeCaller.calls == 0


def test_cache_serves_a_repeated_answer_without_a_second_call(tmp_path, monkeypatch):
    # Two trials, same response text under different keys: the second is a cache
    # hit, so exactly one call happens for the shared text. Serial (workers=1) so
    # the dedup is deterministic -- the cache is best-effort under concurrency,
    # not a single-flight lock, and the test pins the behaviour, not a race.
    with open(os.path.join(tmp_path, "trials.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"trial_key": "a", "response": {"text": "same"}}) + "\n")
        handle.write(json.dumps({"trial_key": "b", "response": {"text": "same"}}) + "\n")
    monkeypatch.setattr(judge_run_mod, "JudgeCaller", FakeCaller)
    FakeCaller.calls = 0
    n = judge_run(str(tmp_path), JUDGE, cache_path=str(tmp_path / "c.sqlite"),
                  verbose=False, workers=1)
    assert n == 2
    assert FakeCaller.calls == 1


def test_default_judge_is_luna_with_reasoning_off(monkeypatch):
    monkeypatch.delenv("BENCH_JUDGE_REASONING_EFFORT", raising=False)
    from bench_v2.judge.caller import DEFAULT_JUDGE_MODEL
    assert DEFAULT_JUDGE_MODEL == "gpt-5.6-luna"
    caps = caps_for(DEFAULT_JUDGE_MODEL)
    assert caps.api == "responses" and caps.reasoning_effort == "none"
    assert caps.temperature is None and caps.seed is None and not caps.logprobs


def test_reasoning_effort_override_only_touches_effort(monkeypatch):
    monkeypatch.delenv("BENCH_JUDGE_REASONING_EFFORT", raising=False)
    base = caps_for("gpt-5.6-luna")                 # reasoning off by default
    assert base.reasoning_effort == "none"
    monkeypatch.setenv("BENCH_JUDGE_REASONING_EFFORT", "high")
    hi = caps_for("gpt-5.6-luna")
    assert hi.reasoning_effort == "high"
    assert (hi.temperature, hi.seed, hi.logprobs, hi.api) == \
           (base.temperature, base.seed, base.logprobs, base.api)
    monkeypatch.delenv("BENCH_JUDGE_REASONING_EFFORT")


def test_gpt54_is_still_reachable_as_the_reproducibility_judge(monkeypatch):
    caps = caps_for("gpt-5.4")
    assert caps.api == "chat" and caps.temperature == 0.0 and caps.logprobs
    monkeypatch.setenv("BENCH_JUDGE_MODEL", "gpt-5.4")
    import importlib

    import bench_v2.tasks.s1_speech.judge_spec as js
    importlib.reload(js)
    assert js.JUDGE.model == "gpt-5.4"
    monkeypatch.delenv("BENCH_JUDGE_MODEL")
    importlib.reload(js)                            # restore for other tests
