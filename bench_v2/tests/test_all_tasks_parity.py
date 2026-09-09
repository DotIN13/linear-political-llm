"""Every ported task's build must be byte-identical to the bench surface it came from.

The test auto-discovers ``bench_v2/tasks/<id>/v1/pilot.py``, imports it, and for
every variant it declares and both conditions builds the same trial through the
old ``bench`` surface and through the pilot, then compares the whole object. This
is the replacement for the generation golden snapshot: it does not hash rendered
prompts, it compares the trial the harness would actually send.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from bench_v2.tests.test_s1_v1_parity import synthetic_item

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def _pilot_ids() -> list[str]:
    return sorted(p.name for p in TASKS_DIR.iterdir()
                  if p.is_dir() and (p / "v1" / "pilot.py").exists())


def _bench_surface(task: str):
    from bench import registry

    registry.load_all()
    try:
        return registry.get_surface(task)()
    except KeyError:
        return None


@pytest.mark.parametrize("task", _pilot_ids())
@pytest.mark.parametrize("condition", ["photos", "no_photos"])
def test_build_matches_bench(task: str, condition: str) -> None:
    surface = _bench_surface(task)
    if surface is None:
        pytest.skip(f"{task}: no bench surface to compare against")
    pilot = importlib.import_module(f"bench_v2.tasks.{task}.v1.pilot")
    item = synthetic_item()

    for variant in pilot.variants():
        where = f"{task} {condition} {variant}"
        bench_trial = surface.build(item, condition, dict(variant), seed=42)
        v2_trial = pilot.build(item, condition, dict(variant), seed=42)

        assert v2_trial.surface == bench_trial.surface, where
        assert v2_trial.condition == bench_trial.condition, where
        assert v2_trial.conversation.messages == bench_trial.conversation.messages, where
        assert v2_trial.conversation.images == bench_trial.conversation.images, where
        assert v2_trial.conversation.sha == bench_trial.conversation.sha, where
        assert v2_trial.variant == bench_trial.variant, where
        assert v2_trial.meta == bench_trial.meta, where
        assert v2_trial.max_new_tokens == bench_trial.max_new_tokens, where
        assert v2_trial.candidates == bench_trial.candidates, where
        assert ([(p.name, p.kind, p.reduce) for p in v2_trial.probe_points]
                == [(p.name, p.kind, p.reduce) for p in bench_trial.probe_points]), where
