"""End-to-end `bench run` + `bench score` on a fake backend. No GPU.

What this pins down:

* **task C** -- an item-invariant condition (E) is run **once** per
  (surface, variant), stored as ``item_id = "__baseline__"``, and broadcast to
  every item by `bench score`, which reports ``outcome_minus_baseline``.
* **task B** -- both A/B orders of the same item are distinct trials rather than
  colliding on one key, and rerunning the command re-does nothing.
* the aggregation rule the whole analysis rests on: variants are repeated
  measures, so ``n_items`` counts items and ``n_rows`` counts rows.
"""

import json
import os

import pytest

from bench import registry
from bench.adaptors.base import BaseAdaptor
from bench.cli import main
from bench.store import write_items
from bench.types import Capability, Response, Trial

N_ITEMS = 6
SURFACES = "vote2020,tea_coffee"
CONDITIONS = "C,E"


@registry.register_adaptor("fake_logprob")
class FakeAdaptor(BaseAdaptor):
    """Deterministic stand-in for local_hf: same interface, arithmetic instead of a model."""

    name = "fake_logprob"
    capabilities = frozenset({Capability.GENERATE, Capability.LOGPROB,
                              Capability.ACTIVATIONS, Capability.IMAGES})

    def __init__(self, model: str = "fake-1", seed: int = 42, **kwargs):
        super().__init__(model=model, seed=seed, **kwargs)
        self.calls = []

    def tokenize(self, text: str):
        return [ord(text)] if len(text) == 1 else [ord(c) for c in text]

    def describe(self):
        base = super().describe()
        base["probe_weights"] = None
        return base

    def run(self, trial: Trial) -> Response:
        self.calls.append((trial.surface, trial.item_id, trial.condition, trial.variant_key))
        n_images = len(trial.conversation.images)
        # a fake dose response keyed on the item index, so the score table and the
        # stratum -> s_img check have something to show
        dose = 0.1 * int(trial.item_id.rsplit("_", 1)[-1]) if n_images else 0.0
        logprobs = {"A": -1.0 + dose, "B": -1.5}
        return Response(
            logprobs=logprobs,
            probe={"probe_id": "fake", "s_txt": 0.2 + dose,
                   "s_img": (0.3 + dose) if n_images else None,
                   "n_image_tokens": 100 * n_images},
            usage={"prefill_tokens": 100, "argmax_token": "A",
                   "top_tokens": [["A", -1.0], ["B", -1.5]]},
            timing_ms=1.0,
        )


@pytest.fixture()
def workspace(tmp_path):
    items = [
        {"item_id": f"lvis3_{i:05d}",
         "images": [f"train2017/{i}_{k}.jpg" for k in range(3)],
         "image_paths": [f"/tmp/{i}_{k}.jpg" for k in range(3)],
         "image_scores": [-0.5 + 0.1 * i] * 3,
         "stratum": i, "primary_iv": "stratum", "split": "explore",
         "covariates": {"n_persons": [0, 0, 0], "has_text_cat_any": False}}
        for i in range(N_ITEMS)
    ]
    path = str(tmp_path / "items.jsonl")
    write_items(path, items)
    return {"items": path, "run": str(tmp_path / "run"), "conv": str(tmp_path / "conv")}


def _run(workspace, *extra):
    return main(["run", "--items", workspace["items"], "--surface", SURFACES,
                 "--conditions", CONDITIONS, "--adaptor", "fake_logprob",
                 "--model", "fake-1", "--out", workspace["run"],
                 "--conversations", workspace["conv"], *extra])


def _rows(workspace):
    with open(os.path.join(workspace["run"], "trials.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_item_invariant_condition_is_run_once_per_variant(workspace, capsys):
    assert _run(workspace) == 0
    capsys.readouterr()
    rows = _rows(workspace)

    baselines = [r for r in rows if r["is_baseline"]]
    images = [r for r in rows if not r["is_baseline"]]

    # 2 surfaces x 2 orders = 4 baseline records, not 2 x 2 x N_ITEMS
    assert len(baselines) == 4
    assert {r["condition"] for r in baselines} == {"E"}
    assert {r["item_id"] for r in baselines} == {"__baseline__"}
    for surface in ("vote2020", "tea_coffee"):
        cell = [r for r in baselines if r["surface"] == surface]
        assert len(cell) == 2
        assert sorted(r["variant"]["order"] for r in cell) == ["ab", "ba"]

    # and the image condition is still fully crossed
    assert len(images) == N_ITEMS * 2 * 2
    assert len(rows) == N_ITEMS * 2 * 2 + 4

    # the saving is exactly the duplication that used to be written out
    naive = N_ITEMS * 2 * 2 * 2
    assert len(rows) == naive - (N_ITEMS - 1) * 2 * 2


def test_the_baseline_conversation_really_is_identical_across_items(workspace):
    """The reason E is run once: every item gives the same bytes."""
    registry.load_all()
    from bench.types import Item, baseline_item

    surface = registry.get_surface("vote2020")()
    variant = {"phrasing": 0, "order": "ab"}
    shas = set()
    with open(workspace["items"], encoding="utf-8") as handle:
        for line in handle:
            item = Item.from_dict(json.loads(line))
            shas.add(surface.build(item, "E", variant).conversation.sha)
    shas.add(surface.build(baseline_item(), "E", variant).conversation.sha)
    assert len(shas) == 1


def test_both_orders_are_separate_trials_and_rerunning_does_nothing(workspace, capsys):
    assert _run(workspace) == 0
    first = _rows(workspace)
    keys = {r["trial_key"] for r in first}
    assert len(keys) == len(first), "no two trials may share a key"

    # ab and ba of the same item are two rows, not one overwriting the other
    for surface in ("vote2020", "tea_coffee"):
        cell = [r for r in first
                if r["surface"] == surface and r["item_id"] == "lvis3_00003"
                and r["condition"] == "C"]
        assert sorted(r["variant"]["order"] for r in cell) == ["ab", "ba"]
        assert len({r["trial_key"] for r in cell}) == 2

    capsys.readouterr()
    assert _run(workspace) == 0
    out = capsys.readouterr().out
    assert _rows(workspace) == first
    assert f"new=0 skipped={len(first)}" in out


def test_orientation_makes_the_two_orders_agree(workspace, capsys):
    """The fake model always prefers letter A, i.e. pure position bias.

    Averaged over the two orders that has to cancel to zero, and the whole of it
    has to show up in position_bias -- which is the point of running both orders.
    """
    assert _run(workspace) == 0
    capsys.readouterr()
    rows = _rows(workspace)
    by_cell = {}
    for row in rows:
        raw = row["outcome"]["extra"]["raw_letter_diff"]
        assert raw > 0, "this fake always prefers whatever sits at letter A"
        expected = 1.0 if row["variant"]["order"] == "ab" else -1.0
        assert row["outcome"]["value"] == pytest.approx(expected * raw)
        by_cell.setdefault((row["surface"], row["item_id"], row["condition"]), []).append(
            row["outcome"]["value"])
    for cell, values in by_cell.items():
        assert len(values) == 2
        assert sum(values) / 2 == pytest.approx(0.0), f"pure position bias must average out: {cell}"
        assert abs(values[0] - values[1]) > 0.5, "and all of it must land in position_bias"


def test_score_broadcasts_the_baseline_to_every_item(workspace, capsys):
    assert _run(workspace) == 0
    capsys.readouterr()
    assert main(["score", "--run", workspace["run"], "--conversations", workspace["conv"]]) == 0
    out = capsys.readouterr().out

    assert "outcome_minus_baseline" not in out  # it is a column header, abbreviated
    assert "minus_base" in out
    assert "position_bias" in out or "pos_bias" in out
    assert "stratum -> s_img" in out
    assert "spearman(stratum, s_img)=+1.0000" in out
    # every image item found a baseline, none went without
    assert f"{N_ITEMS * 2} image items got one, 0 did not" in out
    assert "2 item-invariant cell(s)" in out

    # n counts items, not rows: 2 orders per item must not double it
    for line in out.splitlines():
        if line.startswith("vote2020") and " C " in line:
            fields = line.split()
            assert fields[3] == "1" and fields[4] == "2", line   # n_items=1, n_rows=2


def test_score_by_variant_splits_them_and_filter_selects_one(workspace, capsys):
    assert _run(workspace) == 0
    capsys.readouterr()

    main(["score", "--run", workspace["run"], "--conversations", workspace["conv"],
          "--by", "variant"])
    out = capsys.readouterr().out
    assert '{"order":"ab","phrasing":0}' in out and '{"order":"ba","phrasing":0}' in out

    main(["score", "--run", workspace["run"], "--conversations", workspace["conv"],
          "--filter", "variant.order=ab"])
    out = capsys.readouterr().out
    total = N_ITEMS * 2 * 2 + 4
    assert f"-> {total // 2}/{total} rows" in out
    # one order only: no pair, so no position bias is claimed
    assert "complete ab/ba pairs    : 0" in out


def test_run_can_be_narrowed_to_one_variant(workspace, capsys):
    assert _run(workspace, "--variant", "order=ab") == 0
    capsys.readouterr()
    rows = _rows(workspace)
    assert {r["variant"]["order"] for r in rows} == {"ab"}
    assert len(rows) == N_ITEMS * 2 + 2

    # and a variant that does not exist is refused rather than silently running nothing
    assert _run(workspace, "--variant", "phrasing=7") == 2


def test_a_surface_that_lies_about_item_invariance_is_refused(workspace, capsys):
    """Acting on the claim deletes trials, so the claim is verified against real items."""
    liar = type("Liar", (registry.get_surface("vote2020"),),
                {"name": "liar", "is_item_invariant": lambda self, condition: True})
    registry.register_surface("liar")(liar)
    try:
        assert main(["run", "--items", workspace["items"], "--surface", "liar",
                     "--conditions", "C", "--adaptor", "fake_logprob", "--model", "fake-1",
                     "--out", workspace["run"], "--conversations", workspace["conv"]]) == 2
        assert "item-invariant" in capsys.readouterr().err
    finally:
        registry._SURFACES.pop("liar", None)   # do not leak into the registry listing


def test_manifest_records_both_revisions_and_the_variant_space(workspace, capsys):
    assert _run(workspace) == 0
    capsys.readouterr()
    with open(os.path.join(workspace["run"], "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["measurement_rev"] and len(manifest["measurement_rev"]) == 12
    assert manifest["code_rev"]                       # recorded, but not in the key
    assert manifest["variant_space"]["vote2020"] == [{"phrasing": 0, "order": "ab"},
                                                     {"phrasing": 0, "order": "ba"}]
    assert "bench/surfaces/choice.py" in manifest["measurement_inputs"]
    assert "bench/cli.py" not in manifest["measurement_inputs"]
    assert all(r["ok"] for r in manifest["candidate_gate"])
    assert manifest["n_new"] == manifest["n_planned"] == N_ITEMS * 2 * 2 + 4
