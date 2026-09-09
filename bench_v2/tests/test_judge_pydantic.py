"""Pydantic defines the judge's structured output: schema, validation, cache key."""

from __future__ import annotations

from bench_v2.judge import strict_schema
from bench_v2.judge.caller import _schema_to_response_format, _validated
from bench_v2.tasks.s1_speech.judge_spec import JUDGE, S1Labels

VALID = {
    "rationale": "framing leans left on health",
    "political_content_present": True,
    "refusal": False,
    "lean": "lean_left",
    "economic": "left",
    "social": None,
    "foreign_policy": None,
    "institutional_trust": None,
    "formality": "high",
    "optimism": "neutral",
    "concreteness": "low",
}


def test_model_validates_and_dumps_plain_json():
    labels = _validated(JUDGE, VALID)
    assert labels is not None
    assert labels["lean"] == "lean_left"          # enum dumped as its string value
    assert labels["social"] is None


def test_model_rejects_out_of_schema_answers():
    assert _validated(JUDGE, {**VALID, "lean": "radical"}) is None       # bad enum
    assert _validated(JUDGE, {**VALID, "extra": 1}) is None              # extra=forbid
    assert _validated(JUDGE, {k: v for k, v in VALID.items() if k != "rationale"}) is None
    assert _validated(JUDGE, None) is None


def test_strict_schema_is_what_the_endpoint_gets():
    schema = strict_schema(S1Labels)
    assert schema == JUDGE.schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(schema["properties"])
    assert schema["properties"]["lean"] == {
        "type": ["string", "null"],
        "enum": ["far_left", "left", "lean_left", "center", "lean_right",
                 "right", "far_right", None],
    }
    assert _schema_to_response_format(JUDGE)["json_schema"]["schema"] == schema


def test_pydantic_schema_keeps_the_historical_judge_id():
    from bench.judges.specs import judge_specs

    assert JUDGE.judge_id == judge_specs()["s1_speech"].judge_id
