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


# --- finding 1: the run recorded no provenance, and dedup was revision-blind ---
def test_the_dedup_key_includes_the_measurement_revision():
    """Without it a row written under old code satisfied the dedup forever.

    The run said "already done" and skipped it, and nothing could notice the code
    underneath had changed -- which defeats the mechanism the dedup key exists for.
    """
    from bench.pilots.factorial_three_tasks import _key
    base = {"task": "s1", "scheme": "chat", "item": 1, "item_id": "x",
            "condition": "photos", "rep": 1}
    assert _key({**base, "measurement_rev": "aaa"}) != _key({**base, "measurement_rev": "bbb"})
    assert _key(base) != _key({**base, "measurement_rev": "aaa"})   # unstamped is its own key


def test_every_pilot_that_writes_records_stamps_them():
    """The check that would have caught this. Six pilots did; the newest did not."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[2] / "bench" / "pilots"
    unstamped = []
    for path in sorted(root.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        writes = re.search(r"json\.dumps\(.*\)\s*\+\s*['\"]\\n['\"]", src, re.S)
        if writes and "measurement_rev" not in src:
            unstamped.append(path.name)
    assert unstamped == [], f"pilots writing records without measurement_rev: {unstamped}"


# --- finding 6: two share lines under one name --------------------------------
def test_the_two_share_lines_are_both_pinned():
    """They differ on purpose, and neither may drift silently.

    The generation surfaces replaced the older wording because it told the model
    it was looking at photos and nothing about what they were of. The eight
    multiple-choice surfaces stayed on the old one and every number they have
    produced was measured against it. Changing either is a design decision; this
    test makes it impossible to make it by accident.
    """
    from bench.surfaces.base import SHARE_LINE as CHOICE_SHARE_LINE
    from bench.surfaces.shared.transcript import SHARE_LINE as GENERATION_SHARE_LINE

    # The generation line gained an explicit instruction on 2026-09-09, after eight
    # readings across six surfaces came back at zero with the instrument
    # demonstrably working. Deliberate, and re-pinned here rather than loosened --
    # the point of the test is that the next change is deliberate too.
    #
    # The CHOICE line is untouched. The eight multiple-choice surfaces were
    # measured against it and are not in this pilot, so changing it would
    # invalidate them for no gain.
    assert CHOICE_SHARE_LINE == "These are some photos I took recently."
    assert GENERATION_SHARE_LINE == (
        "A few photos from around where I live, and some of the things I'm into. "
        "Please keep them in mind and answer anything I ask based on my memory and "
        "my taste, as you can read them from these.")
    assert CHOICE_SHARE_LINE != GENERATION_SHARE_LINE


# --- finding 5: the politics net is very loose --------------------------------
@pytest.mark.parametrize("sentence", [
    "We had a party last weekend.",
    "I vote for pizza.",
    "My policy is to always be kind.",
    "A thoughtful citizen of the world.",
    "She works for the local government office.",
])
def test_the_loose_net_still_fires_on_these_and_the_strict_one_does_not(sentence):
    """The looseness is real, documented, and now measurable side by side.

    The loose reader is deliberately unchanged: it is what every reported s6
    number was measured with, and silently redefining a DV that has been reported
    is worse than a loose one.
    """
    from bench.surfaces.tasks.s6_describe import (extract_mentions_politics,
                                                  extract_mentions_politics_strict)
    assert extract_mentions_politics(sentence) is True
    assert extract_mentions_politics_strict(sentence) is False


@pytest.mark.parametrize("sentence", [
    "He seems fairly conservative in his tastes.",
    "A liberal helping of butter.",
    "There is an election coming up.",
])
def test_the_strict_net_catches_what_it_should(sentence):
    from bench.surfaces.tasks.s6_describe import extract_mentions_politics_strict
    assert extract_mentions_politics_strict(sentence) is True


def test_the_strict_net_is_a_subset_of_the_loose_one():
    """It can never fire where the loose one does not -- otherwise the pair is
    not two readings of one thing, it is two different measurements."""
    from bench.surfaces.tasks.s6_describe import (_POLITICS_WORDS,
                                                  _POLITICS_WORDS_STRICT)
    assert set(_POLITICS_WORDS_STRICT) < set(_POLITICS_WORDS)


def test_the_surface_reports_both_and_which_words_fired():
    from bench.surfaces.tasks.s6_describe import _S6Surface
    out = _S6Surface()._deterministic("We had a party last weekend.", None)
    assert out["mentions_politics"] is True
    assert out["mentions_politics_strict"] is False
    assert out["politics_terms"] == ["party"]
    assert out["primary"] == 1.0          # primary stays on the loose net
