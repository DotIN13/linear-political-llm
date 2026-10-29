"""face_swap v4: the best-token swap with every cell eligible. No GPU."""

from __future__ import annotations

from bench_llm import run
from bench_llm.tasks.face_swap.v4 import pilot


def test_items():
    items = pilot.load_items()
    by = {c: sum(1 for it in items if it.data["condition"] == c) for c in pilot.CONDITIONS}
    assert by == {"normal": 40, "top": 240, "random": 240}
    assert len(run.cells(pilot.variants(), items)) == 520 * 4


def test_every_swap_uses_the_whole_frame():
    items = {it.item_id: it for it in pilot.load_items()}
    s = items["top_c2l_p02_s025"].data["splice"]
    assert s["mode"] == "ranked" and s["region"] == "all" and s["direction"] == 1
    assert items["top_l2c_p02_s025"].data["splice"]["direction"] == -1
    assert "splice" not in items["normal_con_p02"].data
    assert pilot.build(items["random_c2l_p02_s025"], pilot.variants()[0]).meta["splice"]["region"] == "all"
