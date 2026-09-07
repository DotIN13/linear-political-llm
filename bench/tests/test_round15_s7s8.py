"""Round 15: the s7/s8 plan, the repeat handle, and the export join."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from bench import registry
from bench.pilots import round15_s7s8 as r15
from bench.types import Item


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    registry.load_all()


def _items(n_per_bucket: int = 6):
    rows = []
    for bucket, base in (("low", -0.7), ("mid", 0.0), ("high", 0.7)):
        for i in range(n_per_bucket):
            rows.append({
                "item_id": f"{bucket}_{i:02d}",
                "images": ["a", "b", "c"],
                "image_paths": ["a.jpg", "b.jpg", "c.jpg"],
                "image_scores": [base, base, base],
                "stratum": {"low": 0, "mid": 5, "high": 9}[bucket],
                "bucket": bucket,
                "split": "explore",
            })
    return rows


# --- the plan ---------------------------------------------------------------
def test_plan_size_is_the_design_multiplied_out():
    """2 tasks x 2 conversations x 12 questions x (18 personas + 1 baseline) x reps."""
    plan = r15.build_plan(_items(), reps=1)
    assert len(plan) == 2 * 2 * 12 * 19, len(plan)
    plan2 = r15.build_plan(_items(), reps=2)
    assert len(plan2) == 2 * len(plan)


def test_both_tasks_both_conversations_all_bands():
    plan = r15.build_plan(_items(), reps=1)
    assert {e["surface"] for e in plan} == set(r15.SURFACE_IDS)
    assert {e["scheme"] for e in plan} == {"chat", "agentic"}
    assert {e["bucket"] for e in plan} == {"low", "mid", "high", "none"}
    # every task asks all twelve of its questions
    for sid in r15.SURFACE_IDS:
        qs = {e["question"] for e in plan if e["surface"] == sid}
        assert len(qs) == 12, (sid, qs)


def test_the_no_photo_baseline_runs_once_per_cell_not_once_per_persona():
    """It is item-invariant -- identical text for every persona -- so running it 18
    times would be 18 copies of one number pretending to be a sample."""
    plan = r15.build_plan(_items(), reps=1)
    base = [e for e in plan if e["condition"] == "E"]
    assert len(base) == 2 * 2 * 12          # task x conversation x question
    assert {e["item_id"] for e in base} == {"no_image"}
    assert all(e["trial"].conversation.images == [] for e in base)


def test_the_photo_arm_carries_the_photos():
    plan = r15.build_plan(_items(), reps=1)
    shown = [e for e in plan if e["condition"] == "C"]
    assert len(shown) == 2 * 2 * 12 * 18
    assert all(len(e["trial"].conversation.images) == 3 for e in shown)


def test_each_record_knows_its_issue():
    plan = r15.build_plan(_items(), reps=1)
    assert {e["domain"] for e in plan} == {"domestic", "foreign"}
    for sid in r15.SURFACE_IDS:
        topics = {e["topic"] for e in plan if e["surface"] == sid}
        assert len(topics) == 12, (sid, topics)


# --- the repeat handle ------------------------------------------------------
def test_reps_produce_distinct_keys_but_identical_conversations():
    """The whole point. Without `rep` in the variant, `trial_key` dedups a second
    reading away as already-done, which is why no earlier round could measure its
    own repeatability -- and an identical condition did give 5.7/5.0/4.3 and then
    3.2/5.0/4.2 with nothing changed.
    """
    from bench.store import trial_key
    surface = registry.get_surface("s7_family_chat")()
    item = Item.from_dict(_items(1)[0])
    keys, shas = set(), set()
    for rep in range(3):
        t = surface.build(item, "C", {"scheme": "chat", "question": "m01", "rep": rep})
        keys.add(trial_key("s7_family_chat", item.item_id, "C", t.variant,
                           "vllm", "m", 1, "rev"))
        shas.add(t.conversation.sha)
    assert len(keys) == 3, "reps must not collapse to one key"
    assert len(shas) == 1, "reps must send a byte-identical conversation"


def test_rep_is_an_int_in_the_variant():
    surface = registry.get_surface("s8_letter_answered")()
    t = surface.build(Item.from_dict(_items(1)[0]), "C",
                      {"scheme": "chat", "question": "c01", "rep": "2"})
    assert t.variant["rep"] == 2


# --- the judge --------------------------------------------------------------
def test_both_tasks_share_one_rubric():
    """s7 and s8 share their twelve issues; scoring them on one rubric is what
    makes 'does the task shape matter' answerable rather than a scale artefact."""
    s7 = registry.get_surface("s7_family_chat")()
    s8 = registry.get_surface("s8_letter_answered")()
    assert s7.judge_spec is not None and s8.judge_spec is not None
    assert s7.judge_spec.judge_id == s8.judge_spec.judge_id


def test_political_score_excludes_the_style_controls():
    labels = {"collective_vs_individual": "left", "public_vs_market": "lean_left",
              "regulation_vs_deregulation": None,
              "liberties_vs_enforcement": "center",
              "formality": "very_high", "optimism": "very_high",
              "concreteness": "very_high",
              "political_content_present": True, "refusal": False}
    out = r15.judge_payload("s7_family_chat", {"labels": labels})
    assert out["political_fields"] == ["collective_vs_individual", "public_vs_market",
                                       "regulation_vs_deregulation",
                                       "liberties_vs_enforcement"]
    # (-2/3 + -1/3 + 0) / 3 = -1/3; the +2.0 style labels must not appear
    assert abs(out["political"] - (-1.0 / 3.0)) < 1e-9, out["political"]
    assert out["values"]["formality"] == 2.0        # recorded, not averaged


def test_judge_missing_and_judge_failed_are_different():
    assert r15.judge_payload("s7_family_chat", None) is None
    assert r15.judge_payload("s7_family_chat", {"error": "429"}) == {"error": "429"}


def test_load_judges_lets_a_rejudge_win(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text(json.dumps({"trial_key": "k", "labels": {"a": 1}}) + "\n"
                 + json.dumps({"trial_key": "k", "labels": {"a": 2}}) + "\n",
                 encoding="utf-8")
    assert r15.load_judges(str(p))["k"]["labels"]["a"] == 2


# --- the smoke slice --------------------------------------------------------
def test_smoke_reaches_both_tasks_both_conversations_and_the_baseline():
    """A prefix of the plan is one task and one conversation, which smokes
    neither the agentic transcript nor s8's inserted exchange."""
    plan = r15.build_plan(_items(), reps=1)
    sl = r15._smoke_slice(plan, 0)
    assert {e["surface"] for e in sl} == set(r15.SURFACE_IDS)
    assert {e["scheme"] for e in sl} == {"chat", "agentic"}
    assert {e["condition"] for e in sl} == {"C", "E"}
    assert {e["bucket"] for e in sl} == {"low", "mid", "high", "none"}
    keys = [(e["surface"], e["scheme"], e["bucket"], e["condition"]) for e in sl]
    assert len(keys) == len(set(keys))
    assert len(sl) == 2 * 2 * 4          # task x conversation x (3 bands + baseline)


# --- the summary the chart reads -------------------------------------------
def test_summary_cells_are_task_by_conversation_by_band():
    recs = []
    for surface in r15.SURFACE_IDS:
        for scheme in ("chat", "agentic"):
            for bucket, pol in (("low", -0.5), ("mid", -0.2), ("high", 0.1)):
                for i in range(3):
                    recs.append({"surface": surface, "scheme": scheme, "bucket": bucket,
                                 "condition": "C", "question": f"q{i}", "rep": 0,
                                 "item_id": f"i{i}", "text": "t", "rule_refusal": False,
                                 "truncated": False, "error": None,
                                 "judge": {"political": pol}})
    s = r15._summarise(recs)
    assert len(s["cells"]) == 2 * 2 * 3
    one = [c for c in s["cells"] if c["surface"] == "s7_family_chat"
           and c["scheme"] == "chat" and c["bucket"] == "low"][0]
    assert one["n"] == 3 and one["n_scored"] == 3
    assert abs(one["political_mean"] - (-0.5)) < 1e-9


def test_summary_reports_repeatability_from_the_texts():
    def rec(rep, text):
        return {"surface": "s7_family_chat", "scheme": "chat", "bucket": "low",
                "condition": "C", "question": "m01", "rep": rep, "item_id": "i0",
                "text": text, "rule_refusal": False, "truncated": False,
                "error": None, "judge": None}
    same = r15._summarise([rec(0, "a"), rec(1, "a")])["repeatability"]
    assert same["cells_with_repeats"] == 1 and same["fraction_identical"] == 1.0
    diff = r15._summarise([rec(0, "a"), rec(1, "b")])["repeatability"]
    assert diff["cells_with_repeats"] == 1 and diff["fraction_identical"] == 0.0
