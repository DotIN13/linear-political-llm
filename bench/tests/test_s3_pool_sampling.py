"""The news-ranking pool becomes 12 topics x 2 sides = 24 candidates, and each
trial shows a balanced sample of 12.

These tests are written against whatever pool is on disk, so they pass on the
six-topic v1 file and will pass on the twelve-topic v2 file without editing --
the properties that matter (one side per topic per trial, a balanced deal, a
per-trial baseline) do not depend on how many topics there are.
"""
from __future__ import annotations

import collections
import json

import pytest

from bench.surfaces.generation import (extract_picks, load_s3_headlines,
                                       sampled_order, shuffled_order)


@pytest.fixture(scope="module")
def pool():
    return load_s3_headlines()


def test_the_pool_is_two_sides_per_topic(pool):
    by_topic = collections.defaultdict(list)
    for h in pool:
        by_topic[h["topic"]].append(h["side"])
    for topic, sides in by_topic.items():
        assert sorted(sides) == ["left", "right"], f"{topic}: {sides}"
    assert len(pool) == 2 * len(by_topic)


def test_every_outlet_is_used_once(pool):
    outlets = [h["outlet"] for h in pool]
    assert len(set(outlets)) == len(outlets), "an outlet used twice confounds slant with source"


def test_a_trial_shows_each_topic_exactly_once(pool):
    n_topics = len({h["topic"] for h in pool})
    for seed in range(40):
        order = sampled_order(pool, "item", seed)
        topics = [pool[i]["topic"] for i in order]
        assert len(order) == n_topics
        assert len(set(topics)) == n_topics, f"seed {seed} repeated or dropped a topic"


def test_the_deal_is_slant_balanced_every_time(pool):
    """Drawing each topic's side independently let one seed deal six right-side
    stories out of six. A lopsided deal inflates or masks a slant preference."""
    n_topics = len({h["topic"] for h in pool})
    seen = collections.Counter()
    for seed in range(200):
        order = sampled_order(pool, "item", seed)
        seen[sum(1 for i in order if pool[i]["side"] == "right")] += 1
    assert len(seen) == 1, f"the deal is not always balanced: {dict(seen)}"
    assert next(iter(seen)) == n_topics // 2


def test_both_sides_of_a_topic_are_shown_across_seeds(pool):
    """Balanced within a trial is not enough -- each topic must appear on both
    sides across trials, or its slant is a constant rather than a variable."""
    sides = collections.defaultdict(set)
    for seed in range(60):
        for i in sampled_order(pool, "item", seed):
            sides[pool[i]["topic"]].add(pool[i]["side"])
    for topic, s in sides.items():
        assert s == {"left", "right"}, f"{topic} only ever shown as {s}"


def test_the_draw_is_deterministic_and_varies_with_seed_and_item(pool):
    assert sampled_order(pool, "a", 1) == sampled_order(pool, "a", 1)
    orders = {tuple(sampled_order(pool, "a", s)) for s in range(20)}
    assert len(orders) > 1, "the seed must change the deal"
    assert sampled_order(pool, "a", 1) != sampled_order(pool, "b", 1)


# --------------------------------------------------------------------------- #
# the dependent variable under sampling
# --------------------------------------------------------------------------- #
def _fake_pool(n_topics: int = 6):
    """Topics with two sides each, slants chosen so the two sides differ in
    magnitude -- true of the real pool, and what makes a naive baseline wrong."""
    rows = []
    for t in range(n_topics):
        lo, hi = -0.40 + 0.05 * t, 0.20 + 0.05 * t
        rows.append({"hid": f"h{2*t+1:02}", "topic": f"t{t}", "side": "left",
                     "outlet": f"L{t}", "slant": lo, "slant_c": lo,
                     "headline": f"left story about topic {t}"})
        rows.append({"hid": f"h{2*t+2:02}", "topic": f"t{t}", "side": "right",
                     "outlet": f"R{t}", "slant": hi, "slant_c": hi,
                     "headline": f"right story about topic {t}"})
    return rows


def _answer(pool, order, pick_positions):
    """An answer that picks the given 1-based positions out of what was shown."""
    return "\n".join(f"{p}. {pool[order[p - 1]]['headline']}" for p in pick_positions)


def test_a_slant_blind_pick_scores_zero_on_the_relative_dv():
    """The point of the correction: 0 whatever the deal happened to be.

    Five picks out of a five-story deal expresses no preference, so the relative
    DV must read exactly 0 even though those five have a non-zero mean slant.
    """
    pool = _fake_pool()
    order = [0, 3, 4, 7, 8]                 # left,right,left,right,left
    r = extract_picks(_answer(pool, order, [1, 2, 3, 4, 5]), pool, order)
    assert r["parse_ok"] and r["n_picked"] == 5
    assert r["slant_rel_mean"] == pytest.approx(0.0)
    assert r["picked_slant_mean"] == pytest.approx(r["slant_shown_mean"])
    assert r["n_shown"] == 5 and r["n_right_shown"] == 2


def test_the_relative_dv_is_immune_to_a_lopsided_deal():
    """Dealt five left-side stories and picking all five is no preference.

    The pool-centred DV reads it as a strong left preference, which is the bug
    the correction exists for.
    """
    pool = _fake_pool()
    lefty = [0, 2, 4, 6, 8]                 # five left versions, nothing else on offer
    r = extract_picks(_answer(pool, lefty, [1, 2, 3, 4, 5]), pool, lefty)
    assert r["parse_ok"]
    assert r["slant_rel_mean"] == pytest.approx(0.0), "no preference was expressible"
    assert r["slant_c_mean"] < -0.2, "the pool-centred DV misreads the deal as a choice"


def test_the_relative_dv_still_detects_a_real_preference():
    """It must not be blind -- picking the left side of every pair scores negative."""
    pool = _fake_pool()
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # both sides of five topics on offer
    lefts = [order.index(i) + 1 for i in (0, 2, 4, 6, 8)]
    r = extract_picks(_answer(pool, order, lefts), pool, order)
    assert r["parse_ok"]
    assert r["slant_rel_mean"] < -0.25, r["slant_rel_mean"]
    assert r["n_right"] == 0


def test_only_shown_topics_can_be_dropped():
    """A topic never dealt was not declined."""
    pool = _fake_pool()
    order = [0, 3, 4, 7, 8, 11]             # six topics on offer, five picked
    r = extract_picks(_answer(pool, order, [1, 2, 3, 4, 5]), pool, order)
    assert r["parse_ok"]
    shown = {pool[i]["topic"] for i in order}
    picked = {pool[pool.index(next(h for h in pool if h["hid"] == hid))]["topic"]
              for hid in r["picked_hids"]}
    assert set(r["dropped_topics"]) == shown - picked
    assert set(r["dropped_topics"]) <= shown, "a topic never shown cannot be dropped"


def test_the_full_permutation_path_is_unchanged(pool):
    """v1 ran on a full shuffle of the pool; that must still behave."""
    order = shuffled_order(pool, "item", 1)
    assert sorted(order) == list(range(len(pool)))
    text = "\n".join(f"{i}. {pool[idx]['headline']}" for i, idx in enumerate(order[:5], 1))
    r = extract_picks(text, pool, order)
    assert r["n_picked"] == 5
    assert r["n_shown"] == len(pool)
    assert r["slant_c_mean"] is not None and r["slant_rel_mean"] is not None


def test_the_relative_dv_does_not_care_which_centring_the_rows_carry():
    """A difference of two means is invariant to a constant shift, so reading
    `slant` or `slant_c` must give the identical answer. That invariance is the
    same one that makes the DV immune to the deal."""
    pool = _fake_pool()
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    lefts = [order.index(i) + 1 for i in (0, 2, 4, 6, 8)]
    text = _answer(pool, order, lefts)
    raw = extract_picks(text, pool, order)["slant_rel_mean"]
    shifted = [dict(h, slant=h["slant"] + 7.25) for h in pool]
    assert extract_picks(text, shifted, order)["slant_rel_mean"] == pytest.approx(raw)
    dropped = [{k: v for k, v in h.items() if k != "slant"} for h in pool]
    assert extract_picks(text, dropped, order)["slant_rel_mean"] == pytest.approx(raw)


# --------------------------------------------------------------------------- #
# What a completed v2 pool has to satisfy. This is deliberately a test rather
# than a note: it fails while the pool is incomplete, so a half-built stimulus
# file cannot quietly become the one an experiment runs on.
# --------------------------------------------------------------------------- #
V2_PATH = "bench/surfaces/tasks/s3_digest/prompts/headlines_v2.jsonl"


def _v2():
    """The pool as it is on disk: jsonl rows plus their sibling header.

    Reads the files directly rather than going through ``load_s3_headlines`` on
    purpose -- this is the gate on the stimulus file itself, so it must fail when the
    file is wrong even if the loader would paper over it.
    """
    import os
    if not os.path.exists(V2_PATH):
        pytest.skip(f"{V2_PATH} not built yet -- 5 of 12 topic pairs still need "
                    f"real coverage; see .tmp/s3-v2-found.json")
    with open(V2_PATH, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    with open(V2_PATH.replace(".jsonl", ".meta.json"), encoding="utf-8") as fh:
        header = json.load(fh)
    return {**header, "headlines": rows}


def test_v2_covers_exactly_the_twelve_topics_of_the_other_two_tasks():
    """The whole point of v2: one topic list across all three tasks."""
    pool = _v2()["headlines"]
    from bench.surfaces.groupchat import load_dataset
    want = {m["topic"] for m in load_dataset()["messages"]}
    got = {h["topic"] for h in pool}
    assert got == want, f"missing {sorted(want - got)}, extra {sorted(got - want)}"
    assert "pentagon" not in got, "pentagon is retired: it has no twin topic"


def test_v2_is_twenty_four_rows_two_sides_per_topic():
    pool = _v2()["headlines"]
    assert len(pool) == 24
    by_topic = collections.defaultdict(list)
    for h in pool:
        by_topic[h["topic"]].append(h["side"])
    assert len(by_topic) == 12
    for topic, sides in by_topic.items():
        assert sorted(sides) == ["left", "right"], f"{topic}: {sides}"


def test_v2_every_row_is_real_and_attributable():
    """No fabricated stimuli. Every row carries a live URL and a rated outlet."""
    pool = _v2()["headlines"]
    for h in pool:
        assert h["url"].startswith("https://"), h["hid"]
        assert h["headline"].strip(), h["hid"]
        assert h["date"], h["hid"]
        assert isinstance(h["slant"], (int, float)), h["hid"]
    outlets = [h["outlet"] for h in pool]
    assert len(set(outlets)) == 24, "an outlet used twice confounds slant with source"
    urls = [h["url"] for h in pool]
    assert len(set(urls)) == 24, "a URL used twice means a row was duplicated"


def test_v2_slant_signs_match_the_side_labels():
    for h in _v2()["headlines"]:
        if h["side"] == "left":
            assert h["slant"] < 0, f"{h['outlet']} is labelled left but rates {h['slant']}"
        else:
            assert h["slant"] > 0, f"{h['outlet']} is labelled right but rates {h['slant']}"


def test_a_story_that_was_not_dealt_cannot_be_picked():
    """The outlet and fuzzy matchers used to scan the whole pool, so a story the
    trial never displayed could be scored as a pick -- a fabricated observation,
    not a parse failure. Restricting them to the deal is the fix."""
    pool = _fake_pool()
    order = [0, 3, 4, 7, 8]                     # five of twelve on screen
    # the answer quotes a story that was NOT dealt, word for word
    unshown = pool[1]["headline"]
    text = "\n".join([_answer(pool, order, [1, 2, 3, 4]), f"5. {unshown}"])
    r = extract_picks(text, pool, order)
    for hid in r["picked_hids"]:
        idx = next(i for i, h in enumerate(pool) if h["hid"] == hid)
        assert idx in order, f"{hid} was never shown but was counted as a pick"
    assert pool[1]["hid"] not in r["picked_hids"]
