"""The phrasing bake-off: the candidates, and the two numbers that decide it."""
from __future__ import annotations
import pytest
from bench import registry
from bench.pilots import probe_s7_phrasing as pb


@pytest.fixture(scope="module", autouse=True)
def _loaded(): registry.load_all()


def test_the_incumbent_and_the_collapse_are_both_in_the_run():
    """A bake-off without the thing you already have tells you nothing, and one
    without the known failure has no floor to measure against."""
    assert "a_short" in pb.PHRASINGS      # round 15's wording, the collapse
    assert "b_committed" in pb.PHRASINGS  # what is on dev right now


def test_every_candidate_shares_the_same_preamble():
    """Only the ask differs. If the message framing moved too, a difference
    between candidates could be either one."""
    msgs = pb.load_messages()
    texts = {n: pb.question_text(msgs["m01"], n) for n in pb.PHRASINGS}
    head = pb.PREAMBLE.format(message=msgs["m01"])
    for n, t in texts.items():
        assert t.startswith(head), n
        assert t[len(head):] == pb.PHRASINGS[n]["tail"], n


def test_candidates_are_distinct():
    tails = [p["tail"] for p in pb.PHRASINGS.values()]
    assert len(set(tails)) == len(tails)


def test_no_candidate_contains_a_political_word():
    """The red line. A candidate that leans the model would win the bake-off by
    cheating -- it would produce opinions we put there."""
    blob = " ".join(p["tail"] for p in pb.PHRASINGS.values()).lower()
    for w in ("politic", "vote", "party", "liberal", "conservative", "democrat",
              "republican", "left-wing", "right-wing", "progressive"):
        assert w not in blob, w


def test_questions_span_round15s_outcomes():
    """Chosen to include the two that collapsed and one that already worked, so a
    candidate cannot look good by being tested only on easy questions."""
    assert set(pb.QUESTION_IDS) == {"m02", "m09", "m01", "m08"}
    assert set(pb.WHY_THESE) == set(pb.QUESTION_IDS)


def test_sides_are_the_extremes_and_do_not_overlap(tmp_path):
    """The real items file lives on the cluster, so this drives the selection
    logic on a synthetic one: it must take the extreme tails, not a random 8."""
    import json
    rows = [{"item_id": f"i{i:02d}", "split": "explore", "images": ["a"],
             "image_paths": ["a.jpg"], "image_scores": [i / 10 - 1.0], "stratum": i}
            for i in range(20)]
    rows += [{"item_id": "skipme", "split": "confirm", "images": ["a"],
              "image_paths": ["a.jpg"], "image_scores": [9.0], "stratum": 0}]
    f = tmp_path / "items.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    left, right = pb.load_sides(str(f), n=4)
    assert len(left) == len(right) == 4
    assert max(r["image_mean"] for r in left) < min(r["image_mean"] for r in right)
    assert not ({r["item_id"] for r in left} & {r["item_id"] for r in right})
    # the extreme tails, and the non-explore row is excluded despite its 9.0
    assert [r["item_id"] for r in left] == ["i00", "i01", "i02", "i03"]
    assert [r["item_id"] for r in right] == ["i16", "i17", "i18", "i19"]


def test_wrapper_detector_catches_the_round15_shape():
    assert pb._is_wrapper("Here's a quick, neutral reply you could send: > ...")
    assert pb._is_wrapper('"That is wild."')
    assert pb._is_wrapper("Maybe something like: ...")
    assert not pb._is_wrapper("That's wild — 50k in one month?")
    assert not pb._is_wrapper("I think it's about time someone did this.")


def test_political_score_uses_only_the_lean_axes():
    p = pb._political({"collective_vs_individual": "left", "public_vs_market": "center",
                       "regulation_vs_freedom": "center", "formality": "very_high"})
    assert abs(p - (-2 / 3 / 3)) < 1e-9
    assert pb._political({"formality": "very_high"}) is None
