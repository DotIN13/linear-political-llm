"""face_swap v2: the items and the trial's splice spec. No GPU."""

from __future__ import annotations

from bench_llm import run
from bench_llm.tasks.face_swap.v2 import pilot


def test_items():
    items = pilot.load_items()
    by = {c: [it for it in items if it.data["condition"] == c] for c in pilot.CONDITIONS}
    assert {c: len(v) for c, v in by.items()} == {"swap": 100, "fill": 100, "normal": 10, "pixel": 100}
    assert len(run.cells(pilot.variants(), items)) == 310 * 4


def test_swap_and_fill_share_the_pair_and_differ_in_mode():
    items = {it.item_id: it for it in pilot.load_items()}
    s, f = items["swap_p03_b07"].data["splice"], items["fill_p03_b07"].data["splice"]
    assert s["mode"] == "swap" and f["mode"] == "fill_check"
    assert s["donor_image"] == f["donor_image"] and s["donor_image"].endswith(".jpg")
    assert items["swap_p03_b07"].image_paths == items["normal_p03"].image_paths


def test_build_passes_the_splice_only_where_there_is_one():
    items = {it.item_id: it for it in pilot.load_items()}
    v = pilot.variants()[2]
    assert pilot.build(items["swap_p00_b00"], v).meta["splice"]["mode"] == "swap"
    assert "splice" not in pilot.build(items["pixel_p00_b00"], v).meta
    assert pilot.build(items["normal_p00"], v).meta["candidates"]
