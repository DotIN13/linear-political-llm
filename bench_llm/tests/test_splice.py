"""bench_llm.splice: the grid, the cell map and the splice plan. No GPU, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from bench_llm import splice as sp
from bench_llm.adaptors import _with_image
from bench_llm.types import Conversation


def test_infer_grid_picks_the_aspect():
    assert sp.infer_grid(19 * 25, 800 / 600) == (25, 19)
    assert sp.infer_grid(14 * 19, 800 / 600) == (19, 14)
    with pytest.raises(ValueError):
        sp.infer_grid(7, 800 / 600)


def test_cells_from_a_mask():
    m = np.zeros((8, 6), bool)
    m[4:, 2:4] = True                      # a block in the bottom middle
    frac = sp.cell_fractions(m, 4, 3)
    assert frac == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    assert sp.person_cells(m, 4, 3)[7] and not sp.person_cells(m, 4, 3)[0]


def test_plans():
    base = [True, False, False, False]     # base: person in cell 0
    donor = [False, True, False, True]     # donor: person in cells 1 and 3
    swap = sp.splice_plan(base, donor, "swap")
    assert [s.kind for s in swap] == ["base", "donor_mean", "donor", "donor_mean"]
    assert swap[2].index == 2
    fill = sp.splice_plan(base, donor, "fill_check")
    assert [s.kind for s in fill] == ["base", "base_mean", "base", "base_mean"]
    assert all(s.kind == "base" for s in sp.splice_plan(base, donor, "normal"))
    assert sp.plan_counts(swap) == {"base": 1, "donor_mean": 2, "donor": 1}
    with pytest.raises(ValueError):
        sp.splice_plan(base, donor[:3], "swap")


def test_the_donor_conversation_only_changes_the_image():
    conv = Conversation(messages=[{"role": "user", "content": [{"type": "image", "image": "/a.jpg"}, {"type": "text", "text": "q"}]}])
    out = _with_image(conv, "/b.jpg")
    assert out.messages[0]["content"][0]["image"] == "/b.jpg" and out.messages[0]["content"][1]["text"] == "q"
    assert conv.messages[0]["content"][0]["image"] == "/a.jpg"
