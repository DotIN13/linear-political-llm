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
    assert P.ARMS == (3, 5, 10)
    assert set(P.ITEMS_FILES) == set(P.ARMS)
    assert list(P.ARMS) == sorted(P.ARMS), "ARMS[0] is the contrast reference"
    # a separate stimulus file per group, so an item of one photo count can never
    # be read as another's
    assert len({P.ITEMS_FILES[a] for a in P.ARMS}) == len(P.ARMS)


def test_questions_are_round_16s_so_they_are_not_a_new_variable():
    assert P.QUESTION_IDS == ["m02", "m09", "m01", "m08"]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_every_photo_attached_exactly_once_in_order(n):
    paths = [f"/p/{i}.jpg" for i in range(n)]
    msgs, tools = build_scheme_messages("agentic", paths, "no_photos", n)
    seen = [part["image"] for m in msgs for part in (m.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image"]
    assert seen == paths, "pixels must appear once each, in the item's own order"
    assert tools is not None


@pytest.mark.parametrize("n", [3, 10])
def test_every_viewed_file_is_opened_from_the_dir_it_was_listed_in(n):
    """A file opened from a path it was never listed under is an inconsistent
    transcript -- the model would be reading a directory that does not contain it."""
    files = files_by_dir(n)
    msgs, _ = build_scheme_messages("agentic", ["x"] * n, "no_photos", n)
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
    msgs, _ = build_scheme_messages("agentic", ["a", "b", "c"], "no_photos", 3)
    assert len(msgs) == 13


def test_ten_photos_costs_turns_and_the_probe_says_so():
    """Photo count and turn count are inseparable here; it must be documented."""
    turns = {n: len(build_scheme_messages("agentic", ["x"] * n, "no_photos", n)[0])
             for n in P.ARMS}
    assert turns == {3: 13, 5: 17, 10: 27}, turns
    # strictly increasing, so the confound cannot be waved away as "roughly equal"
    assert list(turns.values()) == sorted(turns.values())
    assert "13 turns at 3 photos, 17 at 5, 27 at 10" in P.__doc__


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
    msgs, _ = build_scheme_messages("agentic", [], "no_photos", 10)
    imgs = [p for m in msgs for p in (m.get("content") or [])
            if isinstance(p, dict) and p.get("type") == "image"]
    assert imgs == [], "condition E must carry no pixels"
    assert len(msgs) == 27, "but must keep every turn and filename"


def test_run_refuses_an_item_whose_photo_count_disagrees_with_its_arm():
    """The guard that stops a 3-photo item being scored as a 10-photo one."""
    src = open(P.__file__.replace(".pyc", ".py"), encoding="utf-8").read()
    assert "carries {len(paths)} photos, arm expects {arm}" in src


# --------------------------------------------------------------------------- #
# The gate itself. These exist because the first dispatch of round 17 died in
# phase_plan with an UnboundLocalError on the remote: every test above exercised
# the *builders*, none of them ran the phase, and the phase is the thing the
# brief calls a gate. A print statement that crashes is a gate that never fires.
# --------------------------------------------------------------------------- #
def _write_items(path, n_photos, n_per_bucket=6, tokens=380):
    """A synthetic items file shaped like the sampler's real output."""
    import json as _j
    lo_base, hi_base = -0.60, 0.62
    with open(path, "w", encoding="utf-8") as fh:
        for bucket, base in (("lo", lo_base), ("hi", hi_base)):
            for i in range(n_per_bucket):
                drift = 0.01 * i * (-1 if bucket == "lo" else 1)
                scores = [base + drift] * n_photos
                fh.write(_j.dumps({
                    "item_id": f"lvis{n_photos}_{bucket}_{i:05d}",
                    "images": [f"r{j}" for j in range(n_photos)],
                    "image_paths": [f"/img/{bucket}_{i}_{j}.jpg" for j in range(n_photos)],
                    "image_scores": scores,
                    "stratum": 0 if bucket == "lo" else 2,
                    "bucket": "low" if bucket == "lo" else "high",
                    "primary_iv": "bucket", "split": "explore",
                    "covariates": {"num_image_tokens": [tokens] * n_photos},
                }) + "\n")


def test_phase_plan_runs_to_completion_for_both_arms(tmp_path, monkeypatch, capsys):
    """The regression test for the crash: plan must reach the contrast check."""
    files = {}
    for arm in P.ARMS:
        f = tmp_path / f"explore_n{arm}.jsonl"
        _write_items(f, arm)
        files[arm] = str(f)
    monkeypatch.setattr(P, "ITEMS_FILES", files)
    rc = P.phase_plan()
    out = capsys.readouterr().out
    assert rc == 0, out
    # every arm printed its personas, its budget line and its server flag
    for arm in P.ARMS:
        assert f"{arm:2} photos per persona" in out, out
        assert f'"image":{arm}' in out, out
    assert "photo-score contrast check (the TREATMENT)" in out, out
    assert "of the 3-photo gap" in out, out
    # the labelling rule: no bare signed decimal, and no lettered/numbered arms
    assert "photo-score gap" in out, "the treatment scale must name itself"
    assert "most left-looking" in out and "most right-looking" in out, out
    assert "arm" not in out.replace("arms", ""), f"no bare 'arm' labels: {out}"
    assert "Traceback" not in out


def test_phase_plan_warns_when_one_arm_has_a_weaker_contrast(tmp_path, monkeypatch, capsys):
    """The gate's whole purpose: a flattened treatment must not pass silently."""
    files = {}
    for arm in P.ARMS:
        f = tmp_path / f"explore_n{arm}.jsonl"
        # the 10-photo arm gets a deliberately squashed spread
        _write_items(f, arm)
        if arm != P.ARMS[0]:
            import json as _j
            rows = [_j.loads(l) for l in open(f) if l.strip()]
            for r in rows:
                r["image_scores"] = [s * 0.5 for s in r["image_scores"]]
            with open(f, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(_j.dumps(r) + "\n")
        files[arm] = str(f)
    monkeypatch.setattr(P, "ITEMS_FILES", files)
    P.phase_plan()
    out = capsys.readouterr().out
    assert "WARNING" in out, out
    assert "weaker treatment" in out, out


def test_phase_plan_reports_a_missing_items_file_instead_of_crashing(tmp_path, monkeypatch, capsys):
    f = tmp_path / "explore_n3.jsonl"
    _write_items(f, 3)
    files = {3: str(f)}
    for arm in P.ARMS[1:]:
        files[arm] = str(tmp_path / f"nope_{arm}.jsonl")
    monkeypatch.setattr(P, "ITEMS_FILES", files)
    rc = P.phase_plan()
    out = capsys.readouterr().out
    assert rc == 1, "a missing arm must be a non-zero exit, so a script can gate on it"
    assert "ITEMS FILE MISSING" in out
    assert "photos per persona" in out
    assert "--images-per-item 10" in out


def test_phase_plan_flags_a_prompt_that_would_not_fit(tmp_path, monkeypatch, capsys):
    """A truncated prompt means the last photos are never seen."""
    files = {}
    for arm in P.ARMS:
        f = tmp_path / f"explore_n{arm}.jsonl"
        _write_items(f, arm, tokens=900)   # 10 x 900 + 600 + 320 > 8192
        files[arm] = str(f)
    monkeypatch.setattr(P, "ITEMS_FILES", files)
    P.phase_plan()
    out = capsys.readouterr().out
    assert "DOES NOT FIT" in out, out
    assert "simply not seen" in out, out


def test_load_sides_refuses_a_pool_too_small_to_make_both_tails(tmp_path):
    f = tmp_path / "thin.jsonl"
    _write_items(f, 10, n_per_bucket=2)     # 4 items, needs >= 8
    with pytest.raises(ValueError, match="may not support"):
        P.load_sides(str(f))


def test_report_reads_judge_fields_the_rubric_actually_emits():
    """The first run of this pilot printed `opinion 0` for both photo-count groups
    because the report read `states_position`, a field the s2 rubric does not
    have. A nonexistent field reads as absent, which is indistinguishable from
    "the model stated no position in any of 64 answers" -- so this asserts every
    judge field the report touches is one the spec declares.
    """
    from bench.judges import judge_specs
    spec = judge_specs()[P.JUDGE_ID]
    declared = set(spec.label_fields)
    for field in ("political_content_present", "refusal"):
        assert field in declared, f"{field} is not emitted by {P.JUDGE_ID}"
    assert "states_position" not in declared
    # Look for the *usage*, not the word: a field lookup is a quoted string,
    # whereas a comment explaining the bug is prose. Grepping for the bare name
    # made this test fail on its own documentation twice.
    src = open(P.__file__.replace(".pyc", ".py"), encoding="utf-8").read()
    assert '"states_position"' not in src, "the phantom field must not come back"
    assert "'states_position'" not in src, "the phantom field must not come back"
    for field in P.lean_fields():
        assert field in declared, f"{field} is not emitted by {P.JUDGE_ID}"
    assert "regulation_vs_freedom" not in declared, "the conflated axis is gone"


def test_opinion_and_refusal_are_counted_from_the_labels(tmp_path, monkeypatch, capsys):
    """A judged record with a position must be counted as one."""
    import json as _j
    rows = []
    for arm in P.ARMS:
        for i, side in enumerate(["left"] * 2 + ["right"] * 2):
            rows.append({
                "arm": arm, "question": "m01", "side": side,
                "item_id": f"lvis{arm}_{side}_{i}", "image_mean": -0.6 if side == "left" else 0.6,
                "n_photos": arm, "n_turns": 13 if arm == 3 else 27,
                "text": f"answer {arm} {side} {i}",
                "judge": {"labels": {
                    "political_content_present": True, "refusal": False,
                    "collective_vs_individual": "left" if side == "left" else "center",
                    "public_vs_market": "lean_left",
                    "regulation_vs_deregulation": None,
                    "liberties_vs_enforcement": "center",
                }},
            })
    jp = tmp_path / "judged.jsonl"
    jp.write_text("\n".join(_j.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(P, "JUDGED_PATH", str(jp))
    P.phase_report()
    out = capsys.readouterr().out
    assert "opinion rate (n)   3 photos: 4" in out, out
    assert "refusals (n)" in out, out


def test_report_header_is_generated_from_arms_not_hardcoded(tmp_path, monkeypatch, capsys):
    """The header said "THREE PHOTOS vs TEN" and "3 photos is 13 turns, 10 is 27"
    as literals, so adding the 5-photo group left the report describing an
    experiment that was no longer the one being run."""
    import json as _j
    rows = []
    for arm in P.ARMS:
        for i, side in enumerate(["left", "right"]):
            rows.append({"arm": arm, "question": P.QUESTION_IDS[0], "side": side,
                         "item_id": f"i{arm}{i}", "image_mean": -0.6 if side == "left" else 0.6,
                         "n_photos": arm, "n_turns": 0, "text": f"t{arm}{i}",
                         "judge": {"labels": {"political_content_present": True,
                                              "refusal": False,
                                              "collective_vs_individual": "center",
                                              "public_vs_market": "center",
                                              "regulation_vs_deregulation": None,
                                              "liberties_vs_enforcement": "center"}}})
    jp = tmp_path / "j.jsonl"
    jp.write_text("\n".join(_j.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(P, "JUDGED_PATH", str(jp))
    P.phase_report()
    out = capsys.readouterr().out
    for arm in P.ARMS:
        assert f"{arm} photos" in out, f"{arm} missing from the header/table: {out}"
    assert "THREE PHOTOS vs TEN" not in out
    # the turn counts must be computed, so they cannot go stale
    assert "3 photos = 13 turns" in out and "10 photos = 27 turns" in out, out
    assert "5 photos = 17 turns" in out, out


def test_pooled_distinctness_catches_a_cross_group_duplicate(tmp_path, monkeypatch, capsys):
    """The guard that the per-group `distinct` column cannot provide: the same
    answer appearing in two different photo-count groups."""
    import json as _j
    rows = []
    for arm in P.ARMS:
        for i, side in enumerate(["left", "right"]):
            # every left answer is byte-identical across all three groups
            txt = "the same left answer everywhere" if side == "left" else f"unique {arm}"
            rows.append({"arm": arm, "question": P.QUESTION_IDS[0], "side": side,
                         "item_id": f"i{arm}{i}", "image_mean": -0.6 if side == "left" else 0.6,
                         "n_photos": arm, "n_turns": 0, "text": txt,
                         "judge": {"labels": {"political_content_present": True,
                                              "refusal": False,
                                              "collective_vs_individual": "center",
                                              "public_vs_market": "center",
                                              "regulation_vs_deregulation": None,
                                              "liberties_vs_enforcement": "center"}}})
    jp = tmp_path / "j.jsonl"
    jp.write_text("\n".join(_j.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(P, "JUDGED_PATH", str(jp))
    P.phase_report()
    out = capsys.readouterr().out
    assert "duplicates ACROSS groups" in out, out
    assert "distinct 8-word openers" in out, out
