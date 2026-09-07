"""The judge is `gpt-5.6-luna` on the Responses API, and the parameters it is
sent must match what that model will accept.

Every capability asserted here was measured against the live key on 2026-09-07,
one parameter at a time. The tests exist because the failure mode is a hard 400
that stops a whole judging run: sending `temperature` to a model that refuses it
does not degrade, it aborts.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

import bench.judges.caller as C
from bench.judges.specs import (DEFAULT_JUDGE_MODEL, MODEL_CAPS, REASONING_EFFORTS,
                                caps_for, judge_specs)


def test_the_default_judge_is_luna_on_the_responses_api():
    assert DEFAULT_JUDGE_MODEL == "gpt-5.6-luna"
    spec = judge_specs()["s2_proposal"]
    assert spec.model == "gpt-5.6-luna"
    assert spec.api == "responses"


def test_luna_is_sent_no_parameter_it_refuses():
    """temperature, seed, top_p and logprobs are all refused by this model."""
    caps = caps_for("gpt-5.6-luna")
    assert caps.temperature is None, "temperature is refused: 'only the default (1)'"
    assert caps.seed is None, "seed is not a Responses parameter at all"
    assert caps.logprobs is False, "logprobs: 'not supported with this model'"
    assert caps.reasoning_effort in REASONING_EFFORTS


def test_the_old_judge_stays_reachable_and_fully_pinned():
    """gpt-5.4 is the only model on this key that takes all three of strict
    schema, logprobs and temperature=0.0 -- keep it usable for re-checking
    earlier rounds."""
    caps = caps_for("gpt-5.4")
    assert caps.api == "chat"
    assert caps.temperature == 0.0
    assert caps.seed is not None
    assert caps.logprobs is True
    assert caps.reasoning_effort is None


def test_an_unmeasured_model_gets_the_newer_shape():
    """Omitting a knob costs a knob; sending a refused one aborts the run."""
    caps = caps_for("some-model-nobody-has-probed")
    assert caps.api == "responses"
    assert caps.temperature is None and caps.seed is None
    assert caps.logprobs is False


def test_reasoning_effort_offers_only_values_the_endpoint_named():
    assert REASONING_EFFORTS == ("none", "low", "medium", "high", "xhigh", "max")
    # "minimal" is named in one endpoint error string and refused by another,
    # so it must not be offered.
    assert "minimal" not in REASONING_EFFORTS


# --------------------------------------------------------------------------- #
# the request shape, without touching the network
# --------------------------------------------------------------------------- #
class _Recorder:
    """Stands in for the openai client and records the call it was given."""

    def __init__(self) -> None:
        self.responses = self._Responses(self)
        self.chat = self._Chat(self)
        self.calls: List[Dict[str, Any]] = []

    class _Responses:
        def __init__(self, outer: "_Recorder") -> None:
            self.outer = outer

        def create(self, **kwargs: Any) -> Any:
            self.outer.calls.append({"api": "responses", **kwargs})
            return _FakeResponsesReply()

    class _Chat:
        def __init__(self, outer: "_Recorder") -> None:
            self.outer = outer
            self.completions = self

        def create(self, **kwargs: Any) -> Any:
            self.outer.calls.append({"api": "chat", **kwargs})
            return _FakeChatReply()


class _FakeResponsesReply:
    output_text = '{"ok": true}'
    output: List[Any] = []
    usage = None


class _FakeChatReply:
    class _Choice:
        class message:               # noqa: N801 - mimics the SDK's attribute shape
            content = '{"ok": true}'

    usage = None

    def __init__(self) -> None:
        self.choices = [self._Choice()]


def test_the_responses_call_uses_text_format_not_response_format():
    spec = judge_specs()["s2_proposal"]
    client = _Recorder()
    C._request(client, spec, "some answer",
               C._schema_to_response_format(spec), logprobs=True)
    call = client.calls[0]
    assert call["api"] == "responses"
    assert "response_format" not in call, "that is the chat-completions spelling"
    assert "messages" not in call, "Responses takes `input`, not `messages`"
    fmt = call["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["schema"] == spec.schema
    assert "json_schema" not in fmt, "the schema fields are hoisted one level up"


def test_refused_parameters_are_absent_from_the_wire_not_set_to_none():
    """`temperature=None` in a kwargs dict is still a parameter sent as null."""
    spec = judge_specs()["s2_proposal"]
    client = _Recorder()
    C._request(client, spec, "answer", C._schema_to_response_format(spec), logprobs=True)
    call = client.calls[0]
    for refused in ("temperature", "seed", "top_p", "logprobs", "top_logprobs"):
        assert refused not in call, f"{refused} must not be sent to luna at all"
    assert call["reasoning"] == {"effort": spec.reasoning_effort}


def test_the_chat_path_still_pins_everything_for_the_old_model():
    import dataclasses
    spec = dataclasses.replace(judge_specs()["s2_proposal"],
                               model="gpt-5.4", api="chat", temperature=0.0,
                               seed=20260905, logprobs=True, reasoning_effort=None)
    client = _Recorder()
    C._request(client, spec, "answer", C._schema_to_response_format(spec), logprobs=True)
    call = client.calls[0]
    assert call["api"] == "chat"
    assert call["temperature"] == 0.0 and call["seed"] == 20260905
    assert call["logprobs"] is True and call["top_logprobs"] == 5
    assert "reasoning" not in call
    assert call["response_format"]["type"] == "json_schema"


def test_reading_the_answer_skips_reasoning_items():
    """A reasoning model's `output` list holds reasoning entries with no text;
    concatenating them blindly would corrupt the JSON."""

    class _Reasoning:
        type = "reasoning"
        content = None

    class _Piece:
        text = '{"lean": "left"}'

    class _Message:
        type = "message"
        content = [_Piece()]

    class _Reply:
        output_text = ""
        output = [_Reasoning(), _Message()]

    assert C._response_text(_Reply()) == '{"lean": "left"}'


def test_swapping_the_model_changes_the_cache_key():
    """Otherwise a luna verdict would be served from a gpt-5.4 cache row."""
    import dataclasses
    luna = judge_specs()["s2_proposal"]
    old = dataclasses.replace(luna, model="gpt-5.4", api="chat", temperature=0.0,
                              seed=20260905, logprobs=True, reasoning_effort=None)
    assert luna.judge_id != old.judge_id


def test_the_historical_config_keeps_its_historical_cache_key():
    """1,818 judgements are banked under the old ids. Adding capability fields
    must not orphan them, so a spec in the historical shape hashes as before."""
    import dataclasses
    import bench.judges.specs as S
    old = dataclasses.replace(judge_specs()["s2_proposal"], model="gpt-5.4",
                              api="chat", temperature=0.0, seed=S.DEFAULT_SEED,
                              logprobs=True, reasoning_effort=None)
    expected = S.sha256_of({
        "system_prompt": old.system_prompt,
        "schema": old.schema,
        "model": "gpt-5.4",
        "temperature": 0.0,
        "seed": S.DEFAULT_SEED,
    })
    assert old.judge_id == expected, "the pre-existing judge cache must stay reachable"


def test_the_transport_is_part_of_the_cache_key():
    """The same model on two APIs can label differently."""
    import dataclasses
    spec = judge_specs()["s2_proposal"]
    as_chat = dataclasses.replace(spec, api="chat")
    assert spec.judge_id != as_chat.judge_id


def test_every_rubric_picked_up_the_new_model():
    for name, spec in judge_specs().items():
        assert spec.model == DEFAULT_JUDGE_MODEL, name
        assert spec.api == caps_for(DEFAULT_JUDGE_MODEL).api, name
        assert spec.temperature is None, name
