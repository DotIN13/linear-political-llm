"""CLI smoke: the three list/check commands must work with no GPU."""

from bench.cli import main


def test_surfaces_lists_eight(capsys):
    assert main(["surfaces"]) == 0
    out = capsys.readouterr().out
    assert "8 surfaces" in out
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
    assert main(["check", "--surface", "vote2020", "--adaptor", "opencode"]) == 0
    out = capsys.readouterr().out
    assert "DEGRADED" in out and "s_txt" in out


def test_check_json(capsys):
    import json
    main(["check", "--surface", "vote2020", "--adaptor", "local_hf", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "OK" and payload["degraded"] is False


def test_stubs_do_not_crash(capsys):
    assert main(["judge", "--run", "runs/x"]) == 0
    assert main(["report", "--run", "runs/x"]) == 0
    assert "not implemented yet" in capsys.readouterr().out
