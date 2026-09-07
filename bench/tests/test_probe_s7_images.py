"""The 3-vs-10 photo probe: the wording must come from the surface, and the
transcript must scale with the photo count without losing or duplicating pixels.
"""
from __future__ import annotations

import pytest

from bench.pilots import probe_s7_images as P
from bench.surfaces.generation import build_scheme_messages, files_by_dir
from bench.surfaces.groupchat import QUESTION_TEMPLATE


def test_wording_is_the_surfaces_not_a_copy():
    """The one thing that must not drift: this probe tests the shipped wording."""
    q = P.question_text("SOME MESSAGE")
    assert q == QUESTION_TEMPLATE.format(message="SOME MESSAGE")
    assert "what I actually think about it and why" in q
    assert "nothing before or after it" in q


def test_no_political_word_in_the_template():
    """The red line: the persona is the only thing carrying politics."""
    banned = ("liberal", "conservative", "left-wing", "right-wing", "democrat",
              "republican", "progressive", "political", "politics", "ideology")
    low = QUESTION_TEMPLATE.lower()
    assert not [w for w in banned if w in low]


def test_two_arms_only_and_they_differ_in_photo_count():
    assert P.ARMS == (3, 10)
    assert set(P.ITEMS_FILES) == set(P.ARMS)
    # separate stimulus files, so a 10-photo item can never be read as a 3-photo one
    assert P.ITEMS_FILES[3] != P.ITEMS_FILES[10]


def test_questions_are_round_16s_so_they_are_not_a_new_variable():
    assert P.QUESTION_IDS == ["m02", "m09", "m01", "m08"]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_every_photo_attached_exactly_once_in_order(n):
    paths = [f"/p/{i}.jpg" for i in range(n)]
    msgs, tools = build_scheme_messages("agentic", paths, "Q", n)
    seen = [part["image"] for m in msgs for part in (m.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image"]
    assert seen == paths, "pixels must appear once each, in the item's own order"
    assert tools is not None


@pytest.mark.parametrize("n", [3, 10])
def test_every_viewed_file_is_opened_from_the_dir_it_was_listed_in(n):
    """A file opened from a path it was never listed under is an inconsistent
    transcript -- the model would be reading a directory that does not contain it."""
    files = files_by_dir(n)
    msgs, _ = build_scheme_messages("agentic", ["x"] * n, "Q", n)
    opened, listed = [], []
    for m in msgs:
        for call in (m.get("tool_calls") or []):
            fn = (call.get("function") or {})
            arg = fn.get("arguments")
            if isinstance(arg, str):
                import json as _j
                arg = _j.loads(arg)
            path = (arg or {}).get("path")
            if fn.get("name") == "view_image":
                opened.append(path)
            elif fn.get("name") == "list_dir":
                listed.append(path)
    assert listed == [d for d, _n in files], f"listed {listed}"
    expected = [f"{d}/{f}" for d, names in files for f in names]
    assert opened == expected, f"opened {opened}"
    assert len(opened) == n


def test_three_photos_is_unchanged_from_before_the_change():
    """n=3 must be byte-identical to the historical transcript, or round 16's
    numbers stop being comparable with this probe's 3-photo arm."""
    assert files_by_dir(3) == [("/memory/hometown", ["img_0417.jpg", "img_0903.jpg"]),
                              ("/memory/preferences", ["img_3011.jpg"])]
    msgs, _ = build_scheme_messages("agentic", ["a", "b", "c"], "Q", 3)
    assert len(msgs) == 13


def test_ten_photos_costs_turns_and_the_probe_says_so():
    """Photo count and turn count are inseparable here; it must be documented."""
    m3, _ = build_scheme_messages("agentic", ["x"] * 3, "Q", 3)
    m10, _ = build_scheme_messages("agentic", ["x"] * 10, "Q", 10)
    assert len(m3) == 13 and len(m10) == 27
    assert "27 turns" in P.__doc__ and "13" in P.__doc__


def test_both_dirs_are_always_listed():
    """An empty directory listing is a different stimulus from a populated one."""
    for n in range(2, 11):
        files = files_by_dir(n)
        assert len(files) == 2, f"n={n} must use both memory directories"
        assert all(names for _d, names in files), f"n={n} left a directory empty"


def test_pool_is_fixed_strings_not_generated():
    """A transcript has to be reproducible from the file, not from a seed."""
    assert files_by_dir(10) == files_by_dir(10)
    with pytest.raises(ValueError):
        files_by_dir(11)
    with pytest.raises(ValueError):
        files_by_dir(0)


def test_baseline_keeps_the_file_count_when_pixels_are_dropped():
    msgs, _ = build_scheme_messages("agentic", [], "Q", 10)
    imgs = [p for m in msgs for p in (m.get("content") or [])
            if isinstance(p, dict) and p.get("type") == "image"]
    assert imgs == [], "condition E must carry no pixels"
    assert len(msgs) == 27, "but must keep every turn and filename"


def test_run_refuses_an_item_whose_photo_count_disagrees_with_its_arm():
    """The guard that stops a 3-photo item being scored as a 10-photo one."""
    src = open(P.__file__.replace(".pyc", ".py"), encoding="utf-8").read()
    assert "carries {len(paths)} photos, arm expects {arm}" in src
