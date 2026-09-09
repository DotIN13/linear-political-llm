"""Every ported task's reader must agree with the bench surface's ``extract``.

The build parity test proves the prompt is identical. This proves the *reading*
is: the same ``Response`` goes through the old surface's ``extract`` and the new
pilot's ``read``, and the ``kind``/``value``/``extra`` must match. A task whose
text does not parse is still compared -- the failure path is part of the reader.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from bench_v2.tests.test_all_tasks_parity import _bench_surface, _pilot_ids, synthetic_item
from bench_v2.types import Response

PICK_TASKS = {"s9_neighborhood", "s12_explain", "s14_outfits"}
GENERIC_TEXT = (
    "Here's my answer. I'm sorry, but I can't help with campaign material. "
    "Healthcare, climate and housing matter to me; I vote and I am a candidate."
)


def _response_for(task: str, trial) -> list[Response]:
    """One or more responses that exercise the task's reader, parsed and not."""
    if task == "s15_shopping":
        venue = trial.meta["venue_order"][0]
        return [Response(text="", usage={"agent_calls": [
            {"name": "ask_user", "arguments": {"recommended": venue, "reason": "closest"}}]})]
    if task == "s10_groceries":
        name = trial.meta["tool_names"][0]
        return [
            Response(text=f"I'll order from {name}.", usage={}),
            Response(text="", usage={"tool_calls": [name]}),
        ]
    if task == "s11_health":
        return [Response(text="1, 2, 3, 4, 5, 6\nBest first.")]
    if task in PICK_TASKS:
        return [Response(text="1, 2, 3\nReasons.")]
    if task == "s3_digest":
        return [Response(text="1. first\n2. second\n3. third\n4. fourth\n5. fifth")]
    return [Response(text=GENERIC_TEXT)]


@pytest.mark.parametrize("task", _pilot_ids())
def test_read_matches_bench(task: str) -> None:
    surface = _bench_surface(task)
    if surface is None:
        pytest.skip(f"{task}: no bench surface")
    pilot = importlib.import_module(f"bench_v2.tasks.{task}.v1.pilot")
    item = synthetic_item()

    for condition in ("photos", "no_photos"):
        for variant in pilot.variants():
            trial = surface.build(item, condition, dict(variant), seed=42)
            for resp in _response_for(task, trial):
                where = f"{task} {condition} {variant}"
                bench_out = surface.extract(resp, trial)
                v2_out = pilot.read(resp, trial)
                assert v2_out.kind == bench_out.kind, where
                assert v2_out.value == bench_out.value, where
                assert v2_out.extra == bench_out.extra, where
