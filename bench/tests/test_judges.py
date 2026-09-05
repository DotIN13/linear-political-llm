"""Judge layer: JudgeSpec, the schema validator, the cache, and the caller.

No real API is touched here -- the caller is exercised against a fake OpenAI
client, which is exactly the fixture-based path the task prescribes when a key
is unavailable (here the key *is* available, but tests must not depend on it).
"""

import dataclasses
from types import SimpleNamespace

import pytest

from bench.judges import (
    FIVE_MAP, HEDGING_MAP, LEAN_MAP, JudgeCache, JudgeCaller, JudgeSpec,
    judge_specs, response_hash,
)
from bench.judges.caller import _validate

registry = pytest.importorskip("bench.judges.specs")


def test_five_judge_specs_exist_and_s3_has_none():
    specs = judge_specs()
    assert set(specs) == {"s1_speech", "s2_proposal", "s4_bonus", "s5_letter", "s6_describe"}


def test_judge_id_is_stable_and_sensitive():
    specs = judge_specs()
    for sid, spec in specs.items():
        assert spec.judge_id == judge_specs()[sid].judge_id      # stable across reloads
        changed = dataclasses.replace(spec, system_prompt=spec.system_prompt + " x")
        assert changed.judge_id != spec.judge_id                 # prompt is in the id
        changed_model = dataclasses.replace(spec, model="other-model")
        assert changed_model.judge_id != spec.judge_id           # model is in the id


def test_rationale_is_first_then_political_content_present():
    for spec in judge_specs().values():
        props = list(spec.schema["properties"])
        assert props[0] == "rationale"
        assert props[1] == "political_content_present"
        assert spec.schema["additionalProperties"] is False
        assert set(spec.schema["required"]) == set(props)


def test_lean_map_matches_the_board():
    assert LEAN_MAP == {
        "far_left": -1.0, "left": -2 / 3, "lean_left": -1 / 3, "center": 0.0,
        "lean_right": 1 / 3, "right": 2 / 3, "far_right": 1.0,
    }


def test_s1_schema_carries_the_five_dimensions():
    spec = judge_specs()["s1_speech"]
    props = set(spec.schema["properties"])
    assert {"lean", "economic", "social", "foreign_policy", "institutional_trust"} <= props


def test_s4_schema_carries_equality_and_hedging():
    spec = judge_specs()["s4_bonus"]
    props = spec.schema["properties"]
    assert props["hedging"]["enum"] == list(HEDGING_MAP)
    assert "equality_vs_merit" in props


def _spec():
    return JudgeSpec(
        id="t", model="fake", system_prompt="rate it",
        schema={"type": "object",
                "properties": {"rationale": {"type": "string"},
                               "political_content_present": {"type": "boolean"},
                               "lean": {"type": ["string", "null"], "enum": list(LEAN_MAP) + [None]}},
                "required": ["rationale", "political_content_present", "lean"],
                "additionalProperties": False},
        label_map={"lean": LEAN_MAP}, fields=["lean"],
    )


def test_validate_accepts_and_rejects():
    spec = _spec()
    good = {"rationale": "r", "political_content_present": True, "lean": "left"}
    assert _validate(good, spec.schema) is None
    assert _validate({"rationale": "r", "political_content_present": False, "lean": None},
                     spec.schema) is None          # missing tendency -> null, not 0
    assert "missing required" in _validate({"rationale": "r"}, spec.schema)
    assert "not in enum" in _validate({"rationale": "r", "political_content_present": True,
                                       "lean": "bogus"}, spec.schema)
    assert "not a boolean" in _validate({"rationale": "r", "political_content_present": "yes",
                                         "lean": "left"}, spec.schema)


def test_cache_round_trip(tmp_path):
    path = str(tmp_path / "judge.sqlite")
    with JudgeCache(path) as cache:
        assert cache.get(response_hash("hello"), "jid") is None
        cache.put(response_hash("hello"), "jid", {"labels": {"lean": "left"}})
    with JudgeCache(path) as cache:
        assert cache.get(response_hash("hello"), "jid")["labels"]["lean"] == "left"
        # same text, different judge -> miss
        assert cache.get(response_hash("hello"), "jid2") is None


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self._content),
                logprobs=SimpleNamespace(content=[SimpleNamespace(
                    token="left", logprob=-0.1, top_logprobs=[SimpleNamespace(token="right", logprob=-2.0)])]),
            )],
            usage=None,
        )


def test_caller_parses_and_validates_against_a_fake_client(monkeypatch):
    spec = _spec()
    content = '{"rationale": "it argues for redistribution", "political_content_present": true, "lean": "left"}'
    fake = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(content)))

    import bench.judges.caller as caller
    monkeypatch.setattr(caller, "_openai_client", lambda s: fake)
    result = JudgeCaller(spec).call("the answer text")
    assert result["judge_id"] == spec.judge_id
    assert result["labels"]["lean"] == "left"
    assert result["logprobs"][0]["token"] == "left"


def test_caller_retries_in_json_mode_when_schema_is_off(monkeypatch):
    """A bad first payload (out of schema) triggers the JSON-mode retry."""
    spec = _spec()
    good = '{"rationale": "r", "political_content_present": false, "lean": null}'

    class FlakyCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            content = '{"rationale": "r", "lean": "not-an-enum"}' if self.calls == 1 else good
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content), logprobs=None)],
                usage=None)

    completions = FlakyCompletions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    import bench.judges.caller as caller
    monkeypatch.setattr(caller, "_openai_client", lambda s: fake)
    result = JudgeCaller(spec).call("text")
    assert result["labels"]["political_content_present"] is False
    assert completions.calls == 2
