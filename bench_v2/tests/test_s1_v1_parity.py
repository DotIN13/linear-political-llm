"""s1_speech v1 must be a byte-for-byte port of the bench surface.

This is the check that replaces the old generation golden snapshot for this task:
rather than hashing rendered prompts against a file, it builds the same trial
through both trees and compares the objects, and it asserts the judge spec still
hashes to the historical ``judge_id`` (which is what keeps the gpt-5.4 cache
rows reachable).
"""

from __future__ import annotations

import pytest

from bench_v2.tasks.s1_speech.v1 import pilot as v1
from bench_v2.types import Item

BENCH_S1 = "bench/surfaces/tasks/s1_speech/__init__.py"


def synthetic_item() -> Item:
    return Item(
        item_id="t1",
        images=["train2017/a.jpg", "train2017/b.jpg", "train2017/c.jpg"],
        image_paths=["/tmp/lpl/a.jpg", "/tmp/lpl/b.jpg", "/tmp/lpl/c.jpg"],
        image_scores=[0.1, 0.2, 0.3],
        stratum=5,
    )


def bench_surface():
    from bench import registry

    registry.load_all()
    return registry.get_surface("s1_speech")()


@pytest.mark.parametrize("scheme", ["chat", "agentic"])
@pytest.mark.parametrize("condition", ["photos", "no_photos"])
def test_build_matches_bench(condition, scheme):
    surface = bench_surface()
    item = synthetic_item()
    variant = {"scheme": scheme, "question": "q0"}

    bench_trial = surface.build(item, condition, dict(variant), seed=42)
    v2_trial = v1.build(item, condition, dict(variant), seed=42)

    assert v2_trial.surface == bench_trial.surface
    assert v2_trial.condition == bench_trial.condition
    assert v2_trial.conversation.messages == bench_trial.conversation.messages
    assert v2_trial.conversation.images == bench_trial.conversation.images
    assert v2_trial.conversation.sha == bench_trial.conversation.sha
    assert v2_trial.variant == bench_trial.variant
    assert v2_trial.meta == bench_trial.meta
    assert v2_trial.max_new_tokens == bench_trial.max_new_tokens
    assert v2_trial.candidates == bench_trial.candidates
    # ProbePoint is a frozen dataclass defined in each tree, so equality across
    # trees is field equality, not class identity.
    assert ([(p.name, p.kind, p.reduce) for p in v2_trial.probe_points]
            == [(p.name, p.kind, p.reduce) for p in bench_trial.probe_points])


def test_prompt_and_prefill_match_bench():
    from bench.surfaces.shared.prompts import strings, text

    assert v1.PROMPT == text(BENCH_S1)
    assert v1.PREFILL == strings(BENCH_S1, "prefill.json")["prefill"]


def test_judge_spec_matches_bench():
    from bench.judges.specs import judge_specs

    bench_spec = judge_specs()["s1_speech"]
    assert v1.JUDGE.system_prompt == bench_spec.system_prompt
    assert v1.JUDGE.schema == bench_spec.schema
    assert v1.JUDGE.label_map == bench_spec.label_map
    assert v1.JUDGE.fields == bench_spec.fields
    assert v1.JUDGE.judge_id == bench_spec.judge_id


def test_plan_is_cpu_only(capsys):
    assert v1.main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "s1_speech" in out and "trials" in out
