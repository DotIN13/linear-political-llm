"""face_swap v3: the best-token swap items. No GPU."""

from __future__ import annotations

from bench_llm import run
from bench_llm.tasks.face_swap.v3 import pilot


def test_items():
    items = pilot.load_items()
    by = {c: sum(1 for it in items if it.data["condition"] == c) for c in pilot.CONDITIONS}
    assert by == {"normal": 40, "top": 240, "random": 240}
    assert len(run.cells(pilot.variants(), items)) == 520 * 4
    assert len({it.item_id for it in items}) == len(items)


def test_directions_swap_base_and_donor():
    items = {it.item_id: it for it in pilot.load_items()}
    c2l, l2c = items["top_c2l_p03_s010"].data, items["top_l2c_p03_s010"].data
    assert c2l["base"] == l2c["donor"] and c2l["donor"] == l2c["base"]
    assert c2l["splice"]["direction"] == 1 and l2c["splice"]["direction"] == -1
    assert c2l["splice"]["mode"] == "ranked" and c2l["splice"]["share"] == 0.1
    assert items["top_c2l_p03_s010"].image_paths == items["normal_lib_p03"].image_paths


def test_top_and_random_differ_only_in_how():
    items = {it.item_id: it for it in pilot.load_items()}
    t, r = items["top_c2l_p05_s025"].data["splice"], items["random_c2l_p05_s025"].data["splice"]
    assert {k: v for k, v in t.items() if k != "how"} == {k: v for k, v in r.items() if k != "how"}
    v = pilot.variants()[0]
    assert pilot.build(items["top_c2l_p05_s025"], v).meta["splice"]["how"] == "top"
