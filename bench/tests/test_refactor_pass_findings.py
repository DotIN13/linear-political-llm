"""Regression tests for the findings the refactor pass reported and left alone.

Each of these reproduced before the fix. They are here rather than folded into
the existing modules because what they pin is a *specific past defect*, and a
test that says which one it is survives the next person who reorganises this.
"""

import pytest

from bench.surfaces.shared.ordering import sampled_order
from bench.surfaces.tasks.s3_digest import (S3_N_SHOWN, _find_index_markers,
                                            extract_picks, load_s3_headlines)


# --- finding 2: IndexError on a deal of one -----------------------------------
def test_a_one_item_deal_does_not_raise():
    """The fuzzy pass read scored[0] and scored[1] unconditionally.

    Nothing deals one item today, which is what kept it latent -- and is the same
    shape as the order.index() crash that was live last time.
    """
    headlines = load_s3_headlines()
    out = extract_picks("I'd go with the first one.", headlines, [0])
    assert out["parse_ok"] is False        # one pick is not five; it must not guess


def test_a_two_item_deal_still_compares_against_a_runner_up():
    headlines = load_s3_headlines()
    out = extract_picks("1. " + headlines[0]["headline"], headlines, [0, 1])
    assert out["parse_ok"] is False


# --- finding 3: the balanced draw leaned right on an odd topic count ----------
def _odd_pool(n_topics=5):
    return [{"hid": f"h{i}", "topic": f"t{i // 2}",
             "side": "left" if i % 2 == 0 else "right", "slant_c": 0.0}
            for i in range(n_topics * 2)]


def test_the_balanced_draw_no_longer_deals_a_constant_on_an_odd_pool():
    """`half = len(topics) // 2` sent the leftover topic to the else branch,
    which always picks "right": 3 right in 400 of 400 seeds on a 5-topic pool.
    A constant, not a lean.
    """
    pool = _odd_pool(5)
    counts = set()
    for seed in range(400):
        order = sampled_order(pool, f"item{seed}", seed)
        counts.add(sum(1 for i in order if pool[i]["side"] == "right"))
    assert counts == {2, 3}, counts          # both ways the odd topic can fall
    assert len(counts) > 1, "the odd topic is still going the same way every time"


def test_the_odd_topic_goes_both_ways_at_roughly_even_rates():
    pool = _odd_pool(5)
    n_right = [sum(1 for i in sampled_order(pool, f"i{s}", s) if pool[i]["side"] == "right")
               for s in range(400)]
    share_of_3 = sum(1 for x in n_right if x == 3) / len(n_right)
    assert 0.35 < share_of_3 < 0.65, share_of_3


def test_an_even_pool_is_unchanged_and_exactly_balanced():
    """The live pool is 12 topics, so the fix must be a no-op there."""
    pool = _odd_pool(6)
    for seed in range(200):
        order = sampled_order(pool, f"item{seed}", seed)
        assert sum(1 for i in order if pool[i]["side"] == "right") == 3


# --- finding 4: the marker scanner hardcoded 1..12 ----------------------------
def test_markers_above_the_deal_are_dropped_not_mislabelled():
    assert _find_index_markers("13. x  14. y", 12) == []
    assert [n for _o, n in _find_index_markers("13. x  14. y", 15)] == [13, 14]


def test_the_default_bound_is_the_named_constant():
    assert S3_N_SHOWN == 12
    assert _find_index_markers("13. x") == []
    assert [n for _o, n in _find_index_markers("12. x")] == [12]


@pytest.mark.parametrize("text,expected", [
    ("05. a date", []),            # part of a longer numeral
    ("3 Iranian officials", []),   # prose, not a marker
    ("#7 is the one", [7]),
    ("7) also fine", [7]),
])
def test_the_old_marker_discipline_survives_the_widened_pattern(text, expected):
    assert [n for _o, n in _find_index_markers(text)] == expected


# --- finding 7: n_files was unbound on the no_photos branch -------------------
def test_n_files_is_bound_on_the_no_photos_branch():
    """Only short-circuit evaluation kept the meta dict from a NameError.

    Behavioural rather than a source grep: build a real no_photos trial and read
    the field. If the binding goes away again this raises NameError here, which
    is the failure the fix is for.
    """
    from bench import registry
    registry.load_all()
    from bench.types import Item

    surface = registry.get_surface("s1_speech")()
    item = Item(item_id="t", images=[], image_paths=[], image_scores=[], stratum=0)
    trial = surface.build(item, "no_photos", {"scheme": "chat"}, seed=1)
    assert trial.meta["n_files"] == 0
