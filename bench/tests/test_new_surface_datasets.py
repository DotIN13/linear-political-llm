"""Gate for the four new option sets under bench/data/*.jsonl.

These files sit OUTSIDE ``MEASUREMENT_GLOBS`` and outside ``MEASUREMENT_FILES``
(only ``s3_headlines_v2.json`` is named there), so nothing in the harness
notices if one of them is edited. That is exactly the hazard ``letter.py``
solved for s8 with a dataset fingerprint, and it is why these assertions exist:
they are the only thing standing between a hand-edited option and a dependent
variable that quietly means something else.

Every jsonl here is one ``{"record": "meta"}`` line followed by
``{"record": "option"}`` lines. Runnable two ways::

    python -m pytest bench/tests/test_new_surface_datasets.py
    python bench/tests/test_new_surface_datasets.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA = Path(__file__).resolve().parent.parent / "data"

SETS = {
    "s9_neighborhoods_v1.jsonl": "s9_neighborhoods",
    "s12_explain_points_v1.jsonl": "s12_explain_points",
    "s11_health_options_v1.jsonl": "s11_health_options",
    "s13_patch_choice_v1.jsonl": "s13_patch_choice",
}


def load(name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    path = DATA / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["record"] == "meta", f"{name}: first line must be the meta record"
    options = [r for r in rows[1:] if r["record"] == "option"]
    assert len(options) == len(rows) - 1, f"{name}: lines after the first must all be options"
    return rows[0], options


def _slope(x: List[float], y: List[float]) -> float:
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def test_every_set_parses_and_names_itself() -> None:
    for name, expected in SETS.items():
        meta, options = load(name)
        assert meta["set"] == expected, f"{name}: meta says set={meta['set']!r}"
        assert meta["version"], f"{name}: no version"
        assert options, f"{name}: no options"


def test_every_set_states_its_predicted_direction() -> None:
    """Written down before running anything, per the brief. A missing one is a hole."""
    for name in SETS:
        meta, _ = load(name)
        assert meta.get("predicted_direction"), f"{name}: no predicted_direction"
        assert meta.get("dv_primary"), f"{name}: no dv_primary"
        assert meta.get("position_control"), f"{name}: no position_control"


def test_s9_is_a_clean_two_by_five_crossing() -> None:
    meta, options = load("s9_neighborhoods_v1.jsonl")
    assert len(options) == 10 and meta["n_picks"] == 3
    density = [o["density_c"] for o in options]
    comp = [o["composition_c"] for o in options]
    assert abs(sum(density)) < 1e-12, "density_c pool mean is not 0"
    assert abs(sum(comp)) < 1e-12, "composition_c pool mean is not 0"
    # Orthogonal by construction: 5 rungs x 2 levels, fully crossed. If this
    # drifts, the two DVs stop being separable and neither can be read.
    assert abs(sum(d * c for d, c in zip(density, comp))) < 1e-12, "the two codes are correlated"
    assert sorted({d for d in density}) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert sorted({c for c in comp}) == [-1.0, 1.0]
    assert len({o["nid"] for o in options}) == 10
    assert len({o["name"] for o in options}) == 10
    # Length must not carry the code, or "picked the longest" mimics a preference.
    assert len({o["n_words"] for o in options}) == 1, "descriptions differ in word count"
    for o in options:
        low = o["description"].lower()
        for banned in meta["banned_words"]:
            assert banned not in low, f"{o['nid']} contains banned word {banned!r}"


def test_s12_live_and_inert_topics_have_the_same_shape() -> None:
    """The inert topic is only a control if the same DV formula runs on it."""
    meta, options = load("s12_explain_points_v1.jsonl")
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for o in options:
        by_topic.setdefault(o["topic"], []).append(o)
    assert set(by_topic) == {"inflation", "sky_blue"}
    kinds = {t: {o["topic_kind"] for o in rows} for t, rows in by_topic.items()}
    assert kinds["inflation"] == {"live"} and kinds["sky_blue"] == {"inert"}
    for topic, rows in by_topic.items():
        assert len(rows) == 8, f"{topic}: not 8 points"
        codes = [o["code_c"] for o in rows]
        assert abs(sum(codes)) < 1e-12, f"{topic}: code_c pool mean is not 0"
        assert len({o["n_words"] for o in rows}) == 1, f"{topic}: points differ in word count"
        assert abs(_slope(codes, [float(o["n_chars"]) for o in rows])) <= 3.0, \
            f"{topic}: character length tracks the code"
        for o in rows:
            low = o["text"].lower()
            for banned in meta["banned_words"]:
                assert banned not in low, f"{o['pid']} contains banned word {banned!r}"
    # The inert topic must carry no political frame at all.
    assert {o["coding_source"] for o in by_topic["sky_blue"]} == {"none_non_political"}


def test_s11_is_benign_and_offers_no_way_to_delay_care() -> None:
    meta, options = load("s11_health_options_v1.jsonl")
    assert meta["safety"]["both_scenarios_benign"] is True
    assert meta["safety"]["no_option_is_delay_or_do_nothing"] is True
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for o in options:
        by_scenario.setdefault(o["scenario"], []).append(o)
    assert set(by_scenario) == {"sleep", "allergies"}
    routes_seen = []
    for scenario, rows in by_scenario.items():
        assert len(rows) == 6, f"{scenario}: not 6 options"
        codes = [o["institution_c"] for o in rows]
        assert abs(sum(codes)) < 1e-12, f"{scenario}: institution_c pool mean is not 0"
        assert len({o["n_words"] for o in rows}) == 1, f"{scenario}: options differ in word count"
        for o in rows:
            # The red line for this surface, asserted rather than promised.
            assert o["harm_if_delayed"] is False, f"{o['oid']}: harm_if_delayed is not False"
            assert o["worst_outcome"] == "less effective", f"{o['oid']}: worst outcome is not benign"
            low = o["text"].lower()
            for banned in meta["banned_phrases"]:
                assert banned not in low, f"{o['oid']} contains banned phrase {banned!r}"
        routes_seen.append(tuple(o["route"] for o in rows))
    # The same six routes in both scenarios, so route is a constant.
    assert len(set(routes_seen)) == 1, f"the two scenarios offer different routes: {routes_seen}"
    # Exactly one option is flagged as off the left-right axis, and it is the CAM one.
    outside = [o for o in options if o["outside_left_right"]]
    assert {o["route"] for o in outside} == {"cam"}, "outside_left_right must flag cam and only cam"
    assert len(outside) == 2, "one cam option per scenario"
    assert meta.get("ceiling_warning"), "no ceiling warning -- the likeliest failure of this surface"


def test_s13_has_exactly_one_correct_patch_per_bug() -> None:
    """The whole value of the control rests on this. Two right answers and the
    variance it measures is legitimate disagreement rather than instability."""
    meta, options = load("s13_patch_choice_v1.jsonl")
    by_bug: Dict[str, List[Dict[str, Any]]] = {}
    for o in options:
        by_bug.setdefault(o["bid"], []).append(o)
    assert len(by_bug) == meta["n_bugs"] == 4
    for bid, rows in by_bug.items():
        assert len(rows) == 4, f"{bid}: not 4 patches"
        correct = [o for o in rows if o["correct"]]
        assert len(correct) == 1, f"{bid}: {len(correct)} of the four patches are marked correct"
        for o in rows:
            assert o["political"] is False, f"{o['pid']}: marked political"
            assert o["code_c"] == 0.0, f"{o['pid']}: carries a non-zero code"
            if not o["correct"]:
                # Each distractor states how it fails; an unexplained distractor is filler.
                assert o["why_wrong"], f"{o['pid']}: distractor with no stated failure mode"
            else:
                assert o["why_wrong"] is None


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(1 if failures else 0)
