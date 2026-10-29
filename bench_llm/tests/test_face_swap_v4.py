"""face_swap v4: v3's items, every cell eligible. No GPU."""

from __future__ import annotations

from bench_llm.tasks.face_swap.v3 import pilot as v3
from bench_llm.tasks.face_swap.v4 import pilot


def test_only_the_region_changes():
    a = {it.item_id: it for it in v3.load_items()}
    b = {it.item_id: it for it in pilot.load_items()}
    assert a.keys() == b.keys() and len(b) == 520
    s3, s4 = a["top_c2l_p02_s025"].data["splice"], b["top_c2l_p02_s025"].data["splice"]
    assert s4["region"] == "all" and "region" not in s3
    assert {k: v for k, v in s4.items() if k != "region"} == s3
    assert "splice" not in b["normal_con_p02"].data


def test_plan_runs(capsys):
    assert pilot.main(["plan"]) == 0
    assert "[face_swap/v4]" in capsys.readouterr().out
