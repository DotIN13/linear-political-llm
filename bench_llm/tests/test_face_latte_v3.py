"""face_latte v3: pooling the spellings and the log-odds arithmetic. No GPU, no model."""

from __future__ import annotations

import math

from bench_llm import adaptors
from bench_llm.tasks.face_latte.v3 import pilot
from bench_llm.types import Response


def test_log_odds_pools_spellings():
    c = {"latte": {"logprob": math.log(0.2)}, "Latte": {"logprob": math.log(0.4)},
         "coffee": {"logprob": math.log(0.1)}, "Coffee": {"logprob": math.log(0.1)}}
    out = pilot.drink_odds(c)
    assert abs(out["log_odds"] - math.log(0.6 / 0.2)) < 1e-9
    assert abs(out["p_latte"] - 0.75) < 1e-9 and abs(out["mass"] - 0.8) < 1e-9


def test_missing_candidates_give_none():
    assert pilot.drink_odds({})["log_odds"] is None


def test_read_carries_the_odds_and_the_word():
    trial = pilot.build(pilot.load_faces()[0], pilot.variants()[1])
    c = {w: {"logprob": math.log(p)} for w, p in [("latte", .05), ("Latte", .05), ("coffee", .3), ("Coffee", .5)]}
    out = pilot.read(Response(text="Coffee", probe={"score": -0.2, "candidates": c}), trial)
    assert out.extra["label"] == "coffee" and out.value == -0.2
    assert abs(out.extra["log_odds"] - math.log(0.1 / 0.8)) < 1e-9


def test_both_orders_and_candidates_on_the_adaptor():
    assert {v["order"] for v in pilot.variants()} == {"latte_first", "coffee_first"}
    a = adaptors.ADAPTORS["hf_probe"](model="m", model_path="/x/Qwen3-VL-8B-Instruct", model_family="qwen3-vl",
                                      candidates=["latte", "Latte", "coffee", "Coffee"])
    assert a.describe()["candidates"] == ["latte", "Latte", "coffee", "Coffee"]
    assert "candidates" not in adaptors.ADAPTORS["hf_probe"](model="m", model_path="/x/m", model_family="gemma4").describe()


def test_export_keeps_log_odds():
    item = pilot.load_faces()[0]
    v = pilot.variants()[0]
    row = {"item_id": item.item_id, "model": "m", "item": item.data, "error": None, "variant": v,
           "messages": [{"role": "user", "content": [{"type": "image", "image": "/x/a.jpg"}, {"type": "text", "text": "q"}]}],
           "outcome": {"value": -0.1, "extra": {"word": "Latte", "label": "latte", "answer_text": "Latte", "log_odds": 2.0}}}
    assert pilot.export([row], [])[0]["models"]["m"]["latte"]["latte_first"]["log_odds"] == 2.0
