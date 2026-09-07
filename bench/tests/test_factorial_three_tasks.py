"""The three-task factorial: does the plan have the shape the design says?

Driven against a synthetic items file, so the arithmetic and the invariances are
checked here rather than discovered on a cluster with a GPU held open.
"""
from __future__ import annotations

import collections
import json

import pytest

from bench import registry
from bench.pilots import factorial_three_tasks as F


@pytest.fixture(autouse=True)
def _loaded():
    registry.load_all()


def _items_file(tmp_path, per_bucket=10):
    """Three buckets of items, shaped like the sampler's real output."""
    p = tmp_path / "explore.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for bucket, base in (("low", -0.60), ("mid", 0.02), ("high", 0.62)):
            for i in range(per_bucket):
                drift = 0.01 * i * (-1 if bucket == "low" else 1)
                fh.write(json.dumps({
                    "item_id": f"lvis3_{bucket}_{i:05d}",
                    "images": ["a", "b", "c"],
                    "image_paths": [f"/img/{bucket}{i}_{j}.jpg" for j in range(3)],
                    "image_scores": [base + drift] * 3,
                    "stratum": {"low": -1, "mid": 0, "high": 1}[bucket],
                    "bucket": bucket, "primary_iv": "bucket", "split": "explore",
                    "covariates": {"num_image_tokens": [320, 320, 320]},
                }) + "\n")
    return str(p)


@pytest.fixture
def items(tmp_path, monkeypatch):
    path = _items_file(tmp_path)
    monkeypatch.setattr(F, "ITEMS_FILE", path)
    return path


def test_the_three_tasks_are_the_three_on_the_board():
    assert F.TASKS == ("s3_digest", "s7_family_chat", "s8_letter_answered")
    assert F.SCHEMES == ("chat", "agentic")
    assert F.BANDS == ("low", "mid", "high")
    assert F.PER_BAND == 6 and F.N_ITEMS == 12


def test_every_task_contributes_twelve_items(items):
    """The news ranking has one question, so its twelve are deals. Without that
    it would contribute a twelfth of the data and the tasks would not be
    comparable."""
    for task in F.TASKS:
        assert len(F.task_items(task)) == 12, task
    deals = F.task_items("s3_digest")
    assert all(d.startswith("deal") for d in deals), deals
    # and each deal carries its own seed, or they would all be the same deal
    seeds = {F.item_seed("s3_digest", d) for d in deals}
    assert len(seeds) == 12, seeds
    # a question id does not move the deal
    assert F.item_seed("s7_family_chat", "m01") == F.item_seed("s7_family_chat", "m02")


def test_the_plan_is_the_size_the_design_says(items):
    rows = F.plan_rows()
    by = collections.Counter(r["condition"] for r in rows)
    # 3 tasks x 2 conversations x 12 items x 18 personas, + the repeated cell
    repeats = sum(1 for r in rows if r["rep"] == 2)
    assert by["photos"] == 3 * 2 * 12 * 18 + repeats
    # the repeat is one band of one cell, not the whole cell: 216 would buy the
    # same diagnostic for three times the generations
    assert repeats == 12 * F.PER_BAND, repeats
    assert by["photos"] == 1296 + 72
    # the baseline: once per item per task, and NOT once per scheme
    assert by["no_photos"] == 3 * 12, by["no_photos"]
    assert len(rows) == by["photos"] + by["no_photos"]


def test_the_baseline_never_runs_per_persona(items):
    rows = [r for r in F.plan_rows() if r["condition"] == "no_photos"]
    assert {r["item_id"] for r in rows} == {"__baseline__"}
    assert all(r["band"] is None for r in rows)
    # one row per (task, item): the scheme is asked of the surface, not assumed
    per = collections.Counter((r["task"], r["item"]) for r in rows)
    assert set(per.values()) == {1}, dict(per)


def test_the_baseline_scheme_count_comes_from_the_surface(items):
    """If a surface's baseline were scheme-dependent the plan must run both."""
    for task in F.TASKS:
        surface = registry.get_surface(task)()
        assert surface.is_scheme_invariant("no_photos") is True, task


def test_six_personas_per_band_and_the_bands_are_ordered(items):
    personas = F.load_personas()
    assert set(personas) == set(F.BANDS)
    for band in F.BANDS:
        assert len(personas[band]) == 6, band
    import statistics
    m = {b: statistics.fmean(p["image_mean"] for p in personas[b]) for b in F.BANDS}
    assert m["low"] < m["mid"] < m["high"], m
    # no persona appears in two bands
    ids = [p["item_id"] for b in F.BANDS for p in personas[b]]
    assert len(set(ids)) == len(ids) == 18


def test_the_middle_band_is_taken_from_the_centre_not_an_edge(items):
    """The middle band exists to make monotonicity checkable, so it has to sit
    between the ends rather than being the six nearest one of them."""
    personas = F.load_personas()
    mid = [p["image_mean"] for p in personas["mid"]]
    assert all(abs(v) < 0.3 for v in mid), mid


def test_the_repeat_cell_is_one_cell_and_is_distinguishable(items):
    rows = F.plan_rows()
    reps = [r for r in rows if r["rep"] == 2]
    assert {(r["task"], r["scheme"]) for r in reps} == {F.REPEAT_CELL}
    assert {r["band"] for r in reps} == {F.REPEAT_BAND}
    # rep 1 and rep 2 must be different keys or the store would dedup them away
    keys = {F._key(r) for r in rows}
    assert len(keys) == len(rows), "two planned trials share a key"


def test_the_news_ranking_needs_no_judge_and_the_others_do():
    assert "s3_digest" not in F.JUDGE_BY_TASK
    assert F.JUDGE_BY_TASK["s7_family_chat"] == "s2_proposal"
    assert F.JUDGE_BY_TASK["s8_letter_answered"] == "s2_proposal"


def test_the_docstring_records_what_the_design_cannot_separate():
    doc = F.__doc__
    assert "cannot separate" in doc
    assert "turn count" in doc
    assert "scene content" in doc
    assert "fiction we wrote" in doc


def test_a_missing_items_file_is_reported_not_crashed(monkeypatch, capsys):
    monkeypatch.setattr(F, "ITEMS_FILE", "/nowhere/explore.jsonl")
    assert F.phase_plan() == 1
    assert "ITEMS FILE MISSING" in capsys.readouterr().out
