"""s16_bias_detect v3: v2 with the numbering reversed, scored on v2's orientation."""

from __future__ import annotations

import math

from bench_v2.tasks.s16_bias_detect.v2 import pilot as v2
from bench_v2.tasks.s16_bias_detect.v3 import pilot as v3
from bench_v2.types import Response


def test_only_the_numbering_differs_from_v2():
    assert v3.variants() == v2.variants() and v3.DATASET_HASH == v2.DATASET_HASH
    assert (v3.MAX_NEW_TOKENS, v3.TOP_LOGPROBS, v3.MIN_MASS) == (v2.MAX_NEW_TOKENS, v2.TOP_LOGPROBS, v2.MIN_MASS)
    q2, q3 = v2.question("st03b"), v3.question("st03b")
    assert "1 = conservative" in q3 and "5 = liberal" in q3 and "1 = liberal" in q2
    strip = lambda q: [l for l in q.splitlines() if " = " not in l]
    assert strip(q2) == strip(q3)


def test_the_same_meaning_scores_the_same_in_both():
    # "lean liberal" is digit 2 in v2 and digit 4 in v3; both score -0.5.
    for k, name in v3.CATEGORIES.items():
        k2 = next(j for j, n in v2.CATEGORIES.items() if n == name)
        assert v3.VALUE[k] == v2.VALUE[k2]
    a = v2.read(Response(text="2", logprobs={"2": math.log(0.9), "3": math.log(0.1)}))
    b = v3.read(Response(text="4", logprobs={"4": math.log(0.9), "3": math.log(0.1)}))
    assert math.isclose(a.value, b.value) and math.isclose(b.value, -0.45)
