"""s16_bias_detect v4: v3 on all 240 statements; the 18 v1 statements unchanged."""

from __future__ import annotations

from bench_v2.tasks.s16_bias_detect.v3 import pilot as v3
from bench_v2.tasks.s16_bias_detect.v4 import pilot as v4


def test_the_pool_is_all_240_and_keeps_the_18():
    assert len(v4.QUESTION_IDS) == 240 and len(set(v4.QUESTION_IDS)) == 240
    for sid, row in v3.BY_ID.items():
        assert v4.BY_ID[sid]["statement"] == row["statement"]
        assert v4.BY_ID[sid]["text_lean"] == row["text_lean"]
    assert sum(1 for r in v4.BY_ID.values() if r["text_lean"]) == 18


def test_only_the_pool_differs_from_v3():
    assert (v4.MAX_NEW_TOKENS, v4.TOP_LOGPROBS, v4.MIN_MASS) == (v3.MAX_NEW_TOKENS, v3.TOP_LOGPROBS, v3.MIN_MASS)
    assert v4.VALUE == v3.VALUE and v4.SCHEMES == v3.SCHEMES
    assert v4.question("st03b") == v3.question("st03b")
    assert len(v4.variants()) == 3 * 240 * 2
