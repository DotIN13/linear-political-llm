"""Offline checks for the forced-choice pilot: no items file, no model, no GPU."""
from bench.pilots import probe_forced_choice as p

def test_matched_orders_are_identical_across_buckets():
    o = p.matched_orders(10)
    assert len(o) == 12                       # 6 rotations x fwd/rev
    assert o[0][0] == list(range(10))
    assert o[1][0] == list(reversed(range(10)))
    assert o[2][0][0] == 1                    # rotation 1
    for order, _arm in o:                     # every order is a permutation
        assert sorted(order) == list(range(10))

def test_parse_is_strict_and_never_guesses():
    assert p.parse_numbers("3, 7, 1\nbecause...", 3, 10) == [3, 7, 1]
    assert p.parse_numbers("I'd pick 2, 5 and 9.\n", 3, 10) == [2, 5, 9]
    assert p.parse_numbers("Sure! Here are my picks:\n4,4,4\n", 3, 10) is None  # not distinct
    assert p.parse_numbers("It depends on what you value.", 3, 10) is None
    assert p.parse_numbers("", 3, 10) is None
    assert p.parse_numbers("99, 40, 11", 3, 10) is None                        # out of range
    assert p.parse_numbers("2", 1, 4) == [2]

def test_dv_signs_follow_right_c():
    shown = [{"nid": "a", "right_c": 1.0}, {"nid": "b", "right_c": -1.0},
             {"nid": "c", "right_c": 0.0}]
    assert p.dv_from_picks(shown, [1])["right_c_mean"] == 1.0
    assert p.dv_from_picks(shown, [2])["right_c_mean"] == -1.0
    assert p.dv_from_picks(shown, [1, 2, 3])["right_c_mean"] == 0.0

def test_the_pools_load_and_are_the_shape_the_dv_assumes():
    assert len(p.s9_options()) == 10
    assert all("right_c" in r for r in p.s9_options())
    for topic, n in (("inflation", 8), ("sky_blue", 8)):
        rows = p.s12_options(topic)
        assert len(rows) == n, (topic, len(rows))
        assert all("right_c" in r for r in rows)
    bugs = p.s13_bugs()
    assert len(bugs) == 4
    for bid, patches in bugs.items():
        assert len(patches) == 4, bid
        assert sum(1 for x in patches if x["correct"]) == 1, bid
