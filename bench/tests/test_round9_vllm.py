"""The round-9 vLLM pilot, end to end, without a server or a GPU.

That this is possible at all is the point of an HTTP backend: the plan, the
record shape and the export are all exercised here, so the only thing the
cluster adds is the model's actual words.
"""

import json

import pytest

from bench import registry
from bench.pilots import round9_vllm as r9
from bench.types import Capability, Item, Response

registry.load_all()


def _items(n_per_bucket=6):
    rows = []
    for bucket, base in (("low", -0.6), ("mid", 0.05), ("high", 0.65)):
        for i in range(n_per_bucket):
            rows.append({
                "item_id": f"lvis3_{bucket[:2]}_{i:05d}",
                "images": [f"{bucket}/{i}_{j}.jpg" for j in range(3)],
                "image_paths": [f"/tmp/{bucket}_{i}_{j}.jpg" for j in range(3)],
                "image_scores": [base, base + 0.01, base - 0.01],
                "stratum": 0, "split": "explore", "bucket": bucket,
                "covariates": {"categories": ["umbrella"], "n_objects": 3.0},
            })
    return rows


class _Fake:
    """Answers every trial with a canned digest-shaped string."""
    name = "vllm"
    model = "fake"
    capabilities = frozenset({Capability.GENERATE, Capability.IMAGES})

    def __init__(self):
        self.seen = []

    def setup(self):
        pass

    def run(self, trial):
        self.seen.append(trial)
        return Response(text="1. **Slate** — a paraphrase\n2. **HuffPost** — another\n",
                        probe=None, usage={"finish_reason": "stop",
                                           "transcript_shape": "openai_folded"})


def test_plan_is_300_trials_shaped_as_the_design_says():
    """336 -> 300: the P and R arms are gone with the prefill handle.

    Both were prefill contrasts on s1 (P was chat *with*, R was agentic
    *without*), 18 trials each. Prefill is now a property of the surface, applied
    always, so neither is expressible -- and R had already returned 0/18 on the
    9/9-refusal claim it was built to test, which is why the handle went away.
    """
    plan = r9.build_plan(_items())
    assert len(plan) == 300, {k: v for k, v in
                              __import__("collections").Counter(
                                  (p["surface"], p["scheme"], p["arm"]) for p in plan).items()}
    arms = __import__("collections").Counter(p["arm"] for p in plan)
    assert arms["A"] == 252 and arms["C"] == 48
    assert arms["P"] == 0 and arms["R"] == 0


def test_prefill_follows_the_surface_not_the_scheme():
    """It used to be scheme-driven: chat bare, agentic prefilled. It is not any
    more -- a surface either has prefill_text and always uses it, on both schemes,
    or it has none. So s1 now prefills on chat too, which is a real change to what
    s1's chat numbers mean and not just bookkeeping.
    """
    plan = r9.build_plan(_items())
    for p in plan:
        has_text = bool(r9.registry.get_surface(p["surface"])().prefill_text)
        assert p["prefill"] == ("on" if has_text else "off"), (p["surface"], p["scheme"])
    # and it does not vary within a surface any more
    by_surface = {}
    for p in plan:
        by_surface.setdefault(p["surface"], set()).add(p["prefill"])
    assert all(len(v) == 1 for v in by_surface.values()), by_surface


def test_only_s3_gets_a_reversed_order_arm_and_the_reverse_is_exact():
    plan = r9.build_plan(_items())
    with_order = [p for p in plan if p["order_arm"]]
    assert {p["surface"] for p in with_order} == {"s3_digest"}
    assert len(with_order) == 72          # 18 items x 2 schemes x {fwd, rev}
    by_key = {}
    for p in with_order:
        by_key.setdefault((p["item_id"], p["scheme"]), {})[p["order_arm"]] = p["trial"]
    for (item_id, scheme), pair in by_key.items():
        fwd = pair["fwd"].variant["order"]
        rev = pair["rev"].variant["order"]
        assert rev == list(reversed(fwd)), (item_id, scheme)


def test_run_and_export_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(r9, "load_items", lambda: _items(n_per_bucket=1))
    trials = tmp_path / "trials.jsonl"
    monkeypatch.setattr(r9, "TRIALS_PATH", str(trials))
    monkeypatch.setattr(r9, "OUT_DIR", str(tmp_path))

    fake = _Fake()
    written = r9.phase_run(adaptor=fake)
    assert written == len(fake.seen) > 0

    rows = [json.loads(l) for l in trials.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == written
    # every record must say, in the record itself, that the probe is absent
    assert all(r["probe"] is None for r in rows)
    assert all(r["adaptor"] == "vllm" for r in rows)
    assert all(r["transcript_shape"] == "openai_folded" for r in rows)
    # trial_key must be unique -- the fwd/rev order arms are the risky case
    assert len({r["trial_key"] for r in rows}) == len(rows)

    summary = r9.phase_export(trials_path=str(trials), out_dir=str(tmp_path / "up"))
    assert summary["cells"], summary
    assert all(c["s_gen_mean"] is None for c in summary["cells"])
    exported = json.loads((tmp_path / "up" / "all6.json").read_text(encoding="utf-8"))
    assert len(exported) == len(rows)
    assert all(e["s_gen"] is None for e in exported)
    # the s3 extractor ran on the canned text, so picked_positions exists
    s3 = [e for e in exported if e["surface"] == "s3_digest"]
    assert s3 and "picked_positions" in s3[0]["deterministic"]


def test_rerun_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(r9, "load_items", lambda: _items(n_per_bucket=1))
    trials = tmp_path / "trials.jsonl"
    monkeypatch.setattr(r9, "TRIALS_PATH", str(trials))
    monkeypatch.setattr(r9, "OUT_DIR", str(tmp_path))
    first = r9.phase_run(adaptor=_Fake())
    second = r9.phase_run(adaptor=_Fake())
    assert first > 0 and second == 0        # nothing re-done


def test_smoke_slice_covers_agentic_and_the_order_arms():
    """A prefix of the plan is one surface on chat -- useless as a smoke test."""
    plan = r9.build_plan(_items())
    picked = r9._smoke_slice(plan, 4)
    assert len(picked) == 4
    assert {p["scheme"] for p in picked} == {"chat", "agentic"}
    # the hand-written piece of this backend is the agentic fold, so it must be in
    assert any(p["scheme"] == "agentic" for p in picked)
    # and at most one trial per (scheme, arm, order_arm)
    keys = [(p["scheme"], p["arm"], p["order_arm"]) for p in picked]
    assert len(set(keys)) == len(keys)


def test_limit_caps_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(r9, "load_items", lambda: _items(n_per_bucket=1))
    monkeypatch.setattr(r9, "TRIALS_PATH", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(r9, "OUT_DIR", str(tmp_path))
    assert r9.phase_run(adaptor=_Fake(), limit=3) == 3


def test_no_trial_asks_for_a_prefill_the_surface_cannot_give():
    """The bug that killed run 57894892, as an invariant.

    ``PREFILL_BY_SCHEME`` says agentic wants the prefill, but only ``s1_speech``
    was ever given a ``prefill_text``. Asking anyway used to yield
    ``meta["prefill"] = None`` -- a trial recorded as prefill=on that carried no
    prefill, indistinguishable in the data from one that did.
    """
    plan = r9.build_plan(_items())
    for entry in plan:
        text = entry["trial"].meta.get("prefill")
        if entry["prefill"] == "on":
            assert text, f"{entry['surface']}/{entry['scheme']}/{entry['arm']} " \
                         f"says prefill=on but carries no prefill text"
        else:
            assert text is None, f"{entry['surface']}/{entry['scheme']} " \
                                 f"says prefill=off but carries {text!r}"


def test_only_s1_prefills_and_it_does_so_on_both_schemes():
    plan = r9.build_plan(_items())
    on = {e["surface"] for e in plan if e["prefill"] == "on"}
    assert on == {"s1_speech"}, on
    assert {e["scheme"] for e in plan if e["prefill"] == "on"} == {"chat", "agentic"}
    off = {e["surface"] for e in plan if e["prefill"] == "off"}
    assert off == {"s2_proposal", "s3_digest", "s4_bonus",
                   "s5_letter", "s6_describe"}, off


def test_the_prefill_contrast_arms_are_gone():
    """Nothing should be able to ask for prefill on a per-trial basis again --
    that is what let five of six agentic surfaces request a prefill that did not
    exist and run inert for a whole round."""
    plan = r9.build_plan(_items())
    assert not [e for e in plan if e["arm"] in ("P", "R")]
    surface = r9.registry.get_surface("s1_speech")()
    with pytest.raises(ValueError, match="no longer a handle"):
        surface.build(Item.from_dict(_items()[0]), "C", {"scheme": "chat", "prefill": "off"})


def test_the_smoke_slice_reaches_every_surface():
    """A prefix-keyed smoke was all s1, which is how the missing prefill_text got
    past a green smoke and killed the full run 80 records in."""
    plan = r9.build_plan(_items())
    sl = r9._smoke_slice(plan, 0)
    assert {e["surface"] for e in sl} == set(r9.SURFACE_IDS)
    # and one per (surface, scheme, arm, order_arm), no duplicates
    keys = [(e["surface"], e["scheme"], e["arm"], e["order_arm"]) for e in sl]
    assert len(keys) == len(set(keys))
    # 28 -> 26: the two s1 prefill-contrast arms no longer contribute a cell.
    assert len(sl) == 26


def test_no_surface_is_left_at_the_inherited_400_token_cap():
    """s2 truncated 17/18 at 400. The cap is a design choice per task, so it has
    to be *chosen* per task rather than inherited from `_make`'s default."""
    caps = {sid: r9.registry.get_surface(sid)().max_new_tokens for sid in r9.SURFACE_IDS}
    assert caps == {"s1_speech": 1400, "s2_proposal": 1200, "s5_letter": 800,
                    "s3_digest": 900, "s6_describe": 600, "s4_bonus": 1000}, caps


def test_smoke_phase_passes_limit_zero_through(monkeypatch):
    """`--limit 0` must reach `_smoke_slice` as 0, not become 4.

    `phase_run(limit=args.limit or 4, smoke=True)` made `smoke 0` run four
    s1-only trials -- the prefix behaviour the surface-keyed slice exists to
    remove -- while the unit test called `_smoke_slice(plan, 0)` directly and
    stayed green. Round 11 had to run `smoke 28` to work around it.
    """
    seen = {}
    monkeypatch.setattr(r9, "phase_run",
                        lambda **kw: seen.update(kw) or 0)
    monkeypatch.setattr("sys.argv", ["round9_vllm", "--phase", "smoke", "--limit", "0"])
    r9.main()
    assert seen == {"limit": 0, "smoke": True}, seen


# --- the judge join --------------------------------------------------------
def test_judge_payload_averages_only_the_political_fields():
    """`political` must be the left/right axes, not the style controls.

    formality / optimism / concreteness are on a different five-point map and
    exist to show that something did *not* move with the photo. Folding them
    into the political score would dilute the effect with the controls designed
    to be flat.
    """
    from bench.judges.specs import judge_specs
    spec = judge_specs()["s1_speech"]
    labels = {f: "center" for f in spec.fields if spec.label_map.get(f) is not None
              and "formality" not in f}
    labels.update({"lean": "lean_right", "economic": "right", "social": "center",
                   "foreign_policy": "center", "institutional_trust": "center",
                   "formality": "very_high", "optimism": "very_high",
                   "concreteness": "very_high"})
    out = r9.judge_payload("s1_speech", {"labels": labels})
    assert out["political_fields"] == ["lean", "economic", "social",
                                       "foreign_policy", "institutional_trust"]
    # (1/3 + 2/3 + 0 + 0 + 0) / 5 = 0.2 -- the +2.0 style labels must not appear
    assert abs(out["political"] - 0.2) < 1e-9, out
    assert out["values"]["formality"] == 2.0      # still recorded, just not averaged


def test_judge_payload_distinguishes_failed_from_never_asked():
    assert r9.judge_payload("s1_speech", None) is None
    err = r9.judge_payload("s1_speech", {"error": "HTTP 429"})
    assert err == {"error": "HTTP 429"}
    assert r9.judge_payload("s1_speech", {"labels": {}})["error"]


def test_load_judges_lets_a_rejudge_win(tmp_path):
    p = tmp_path / "judges.jsonl"
    p.write_text(
        json.dumps({"trial_key": "k1", "labels": {"lean": "center"}}) + "\n"
        + json.dumps({"trial_key": "k1", "labels": {"lean": "left"}}) + "\n"
        + json.dumps({"labels": {"lean": "right"}}) + "\n",       # no key: ignored
        encoding="utf-8")
    got = r9.load_judges(str(p))
    assert set(got) == {"k1"}
    assert got["k1"]["labels"]["lean"] == "left"


def test_load_judges_is_empty_when_the_judge_never_ran(tmp_path):
    assert r9.load_judges(str(tmp_path / "nope.jsonl")) == {}
