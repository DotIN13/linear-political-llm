"""CLI smoke: the list/check commands must work with no GPU.

`bench check` now loads the model by default, because the candidate gate does a
real forward pass -- so every check here passes --no-candidates. The candidate
gate itself is exercised on the GPU in bench/smoke.sbatch.
"""

import pytest

from bench.cli import build_parser, dotted_get, main, matches, parse_constraints


def test_surfaces_lists_eight(capsys):
    assert main(["surfaces"]) == 0
    out = capsys.readouterr().out
    assert "8 surfaces" in out
    assert "Answer with a single letter." in out       # options surface, not word candidates
    for name in ["vote2020", "guns", "healthcare", "border",
                 "tea_coffee", "cat_dog", "beach_mountain", "morning_night"]:
        assert name in out
    assert "requires" in out


def test_adaptors_lists_both(capsys):
    assert main(["adaptors"]) == 0
    out = capsys.readouterr().out
    assert "local_hf" in out and "opencode" in out
    assert "activations" in out


def test_check_reports_degradation(capsys):
    assert main(["check", "--surface", "vote2020", "--adaptor", "opencode",
                 "--no-candidates"]) == 0
    out = capsys.readouterr().out
    assert "DEGRADED" in out and "s_txt" in out


def test_check_json(capsys):
    import json
    main(["check", "--surface", "vote2020", "--adaptor", "local_hf",
          "--no-candidates", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["capabilities"]["status"] == "OK"
    assert payload["capabilities"]["degraded"] is False
    assert payload["variants"]["problems"] == []
    assert payload["variants"]["canonical"] == ['{"order":"ab","phrasing":0}',
                                                '{"order":"ba","phrasing":0}']


def test_check_prints_the_declared_variant_space(capsys):
    assert main(["check", "--surface", "vote2020,tea_coffee", "--adaptor", "local_hf",
                 "--no-candidates"]) == 0
    out = capsys.readouterr().out
    assert out.count("variants (2)") == 2


def test_candidate_gate_is_on_by_default_in_both_check_and_run():
    """It is a gate, so it has to be the default; the flags only turn it off."""
    parser = build_parser()
    assert parser.parse_args(["check", "--surface", "s", "--adaptor", "a"]).candidates is True
    assert parser.parse_args(["check", "--surface", "s", "--adaptor", "a",
                              "--no-candidates"]).candidates is False
    run = parser.parse_args(["run", "--items", "i", "--surface", "s", "--out", "o"])
    assert run.candidate_gate is True
    assert run.variant is None            # the surface's declared space, unnarrowed


def test_dotted_path_filter():
    row = {"surface": "vote2020", "variant": {"phrasing": 0, "order": "ab"},
           "outcome": {"value": 1.5}}
    assert dotted_get(row, "variant.phrasing") == 0
    assert dotted_get(row, "variant.nope") is None
    assert matches(row, parse_constraints(["variant.phrasing=0"]))
    assert not matches(row, parse_constraints(["variant.phrasing=1"]))
    assert matches(row, parse_constraints(["variant.order=ab,surface=vote2020"]))
    assert not matches(row, parse_constraints(["variant.order=ba"]))
    # the same constraints applied to a bare variant dict, which is how `run` uses it
    assert matches({"phrasing": 0, "order": "ab"},
                   parse_constraints(["phrasing=0"]))
    with pytest.raises(ValueError):
        parse_constraints(["phrasing"])


def test_stubs_do_not_crash(capsys):
    assert main(["judge", "--run", "runs/x"]) == 0
    assert main(["report", "--run", "runs/x"]) == 0
    assert "not implemented yet" in capsys.readouterr().out
