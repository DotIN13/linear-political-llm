"""face_swap v1: items, the four asks, the read. No GPU, no model."""

from __future__ import annotations

import math

from bench_llm import run
from bench_llm.tasks.face_swap.v1 import pilot
from bench_llm.types import Response


def test_items_and_trials():
    items = pilot.load_items()
    assert len(items) == 110 and sum(1 for it in items if it.data["kind"] == "self") == 10
    assert len(run.cells(pilot.variants(), items)) == 440


def test_the_four_asks():
    it = pilot.load_items()[0]
    q = {(v["question"], v["order"]): pilot.build(it, v).meta["question"] for v in pilot.variants()}
    assert q[("lean", "liberal_first")] == ("Do you think this person is more likely to be liberal leaning or "
                                            "conservative leaning? Answer with one word.")
    assert q[("lean", "conservative_first")].startswith("Do you think this person is more likely to be conservative leaning or liberal")
    assert q[("latte", "latte_first")] == "Take a guess, do you think I like latte or coffee? Answer with one word."
    assert q[("latte", "coffee_first")] == "Take a guess, do you think I like coffee or latte? Answer with one word."


def test_each_trial_scores_only_its_own_words():
    it = pilot.load_items()[0]
    for v in pilot.variants():
        words = pilot.build(it, v).meta["candidates"]
        assert len(words) == 4 and (("coffee" in words) == (v["question"] == "latte"))


def test_read_gives_word_probe_and_log_odds_on_the_conservative_coded_side():
    it = pilot.load_items()[0]
    lean_v = [v for v in pilot.variants() if v["question"] == "lean"][0]
    latte_v = [v for v in pilot.variants() if v["question"] == "latte"][0]
    c = {w: {"logprob": math.log(p)} for w, p in [("conservative", .1), ("Conservative", .1), ("liberal", .4), ("Liberal", .4)]}
    out = pilot.read(Response(text="Liberal.", probe={"score": -0.3, "candidates": c}), pilot.build(it, lean_v))
    assert out.extra["label"] == "liberal" and out.value == -0.3 and abs(out.extra["log_odds"] - math.log(0.2 / 0.8)) < 1e-9
    c = {w: {"logprob": math.log(p)} for w, p in [("coffee", .3), ("Coffee", .3), ("latte", .2), ("Latte", .2)]}
    out = pilot.read(Response(text="Coffee", probe={"score": 0.1, "candidates": c}), pilot.build(it, latte_v))
    assert out.extra["label"] == "coffee" and abs(out.extra["log_odds"] - math.log(0.6 / 0.4)) < 1e-9
    assert pilot.read(Response(text="I cannot say.", probe={}), pilot.build(it, latte_v)).extra["label"] == "other"
