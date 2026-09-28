"""s16_bias_detect v2: five categories, read from the first token's probabilities."""

from __future__ import annotations

import math

from bench_v2.tasks.s16_bias_detect.v1 import pilot as v1
from bench_v2.tasks.s16_bias_detect.v2 import pilot as v2
from bench_v2.types import Response


def lp(**probs):
    return {k: math.log(v) for k, v in probs.items()}


def test_same_design_as_v1_but_the_ask():
    assert v2.BY_ID == v1.BY_ID and v2.DATASET_HASH == v1.DATASET_HASH
    assert v2.variants() == v1.variants()
    q = v2.question("st03b")
    for k, name in v2.CATEGORIES.items():
        assert f"{k} = {name}" in q
    assert v1.BY_ID["st03b"]["statement"] in q


def test_five_categories_map_onto_v1s_scale():
    assert v2.VALUE == {"1": -1.0, "2": -0.5, "3": 0.0, "4": 0.5, "5": 1.0}


def test_score_moves_when_the_top_digit_does_not():
    # Both replies write "1"; the second puts more weight on "2". v1 would read
    # them identically; v2 separates them.
    a = v2.read(Response(text="1", logprobs=lp(**{"1": 0.98, "2": 0.02})))
    b = v2.read(Response(text="1", logprobs=lp(**{"1": 0.80, "2": 0.20})))
    assert a.extra["top"] == b.extra["top"] == "1"
    assert a.value < b.value
    assert math.isclose(a.value, -0.99)


def test_variants_of_a_digit_token_pool_and_others_are_ignored():
    out = v2.read(Response(text="3", logprobs={"3": math.log(0.5), " 3": math.log(0.3),
                                               "30": math.log(0.1), "The": math.log(0.1)}))
    assert math.isclose(out.extra["mass"], 0.8)
    assert out.extra["probs"]["3"] == 1.0 and out.value == 0.0


def test_a_reply_that_does_not_start_with_a_number_is_unread():
    out = v2.read(Response(text="Based on your memory, 2",
                           logprobs=lp(Based=0.9, **{"2": 0.05})))
    assert out.value is None and out.extra["mass"] < v2.MIN_MASS
    assert out.extra["written"] == "2"


def test_no_logprobs_is_unread():
    assert v2.read(Response(text="4")).value is None


def test_the_run_asks_the_server_for_logprobs():
    src = (v2.HERE / "pilot.py").read_text()
    assert '"top_logprobs": TOP_LOGPROBS' in src and v2.TOP_LOGPROBS == 20
