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


def test_the_default_judge_is_gpt_5_4_and_fully_pinned():
    """Not the newest model, on purpose.

    gpt-5.4 is the only one this key can reach that takes strict json_schema and
    logprobs and temperature=0.0 together, so it is the only judge whose verdicts
    are reproducible. Every model past it refuses `temperature`. Since every
    round's numbers are compared against every other round's, a reproducible
    judge beats a cleverer one.
    """
    assert DEFAULT_JUDGE_MODEL == "gpt-5.4"
    spec = judge_specs()["s2_proposal"]
    assert spec.model == "gpt-5.4"
    assert spec.api == "chat"
    assert spec.temperature == 0.0, "greedy decoding must be pinned"
    assert spec.seed is not None
    assert spec.logprobs is True
    assert spec.reasoning_effort is None


def test_the_newer_models_stay_usable_for_re_running_the_comparison():
    """Reverting the default must not delete the capability work."""
    for model in ("gpt-5.6-luna", "gpt-5.5", "gpt-6-astra"):
        caps = caps_for(model)
        assert caps.api == "responses", model
        assert caps.reasoning_effort in REASONING_EFFORTS, model


def test_the_model_can_be_overridden_without_an_edit(monkeypatch):
    """`judge_specs()` reads the env var per call, so no module reload is needed.

    Do not reload `bench.judges.specs` to test this: a reload rebinds `LEAN_MAP`
    to a fresh object, and code elsewhere identifies the lean axes with
    `label_map.get(f) is LEAN_MAP`. That silently broke two unrelated tests when
    this test did reload.
    """
    monkeypatch.setenv("BENCH_JUDGE_MODEL", "gpt-5.6-luna")
    spec = judge_specs()["s2_proposal"]
    assert spec.model == "gpt-5.6-luna"
    assert spec.api == "responses"
    assert spec.temperature is None
    assert spec.reasoning_effort == "high"


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


def _luna_spec():
    """A spec on the newer model, whatever the default happens to be."""
    import dataclasses
    caps = caps_for("gpt-5.6-luna")
    return dataclasses.replace(
        judge_specs()["s2_proposal"], model="gpt-5.6-luna", api=caps.api,
        temperature=caps.temperature, seed=caps.seed, logprobs=caps.logprobs,
        reasoning_effort=caps.reasoning_effort)


def test_the_responses_call_uses_text_format_not_response_format():
    spec = _luna_spec()
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
    spec = _luna_spec()
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
    luna = _luna_spec()
    pinned = judge_specs()["s2_proposal"]
    assert pinned.model == "gpt-5.4"
    assert luna.judge_id != pinned.judge_id


def test_the_historical_config_keeps_its_historical_cache_key():
    """1,818 judgements are banked under the old ids. Adding capability fields
    must not orphan them, so a spec in the historical shape hashes as before."""
    import dataclasses
    import bench.judges.specs as S
    old = judge_specs()["s2_proposal"]   # the default is this shape again
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
    spec = _luna_spec()
    as_chat = dataclasses.replace(spec, api="chat")
    assert spec.judge_id != as_chat.judge_id


def test_every_rubric_uses_the_default_model_and_its_measured_capabilities():
    caps = caps_for(DEFAULT_JUDGE_MODEL)
    for name, spec in judge_specs().items():
        assert spec.model == DEFAULT_JUDGE_MODEL, name
        assert spec.api == caps.api, name
        assert spec.temperature == caps.temperature, name
        assert spec.seed == caps.seed, name
        assert spec.logprobs == caps.logprobs, name
        assert spec.reasoning_effort == caps.reasoning_effort, name


# --------------------------------------------------------------------------- #
# The axis fix. `regulation_vs_freedom` conflated economic regulation with state
# coercive power, which run in opposite directions politically -- so an answer
# opposing an immigration raid scored `lean_right`. Both gpt-5.4 and
# gpt-5.6-luna gave that label independently, which is what established it as a
# rubric fault rather than a judge fault.
# --------------------------------------------------------------------------- #
AFFECTED = ("s2_proposal", "s5_letter")


@pytest.mark.parametrize("rubric", AFFECTED)
def test_the_conflated_axis_is_gone(rubric):
    spec = judge_specs()[rubric]
    assert "regulation_vs_freedom" not in spec.fields
    assert "regulation_vs_freedom" not in spec.schema["properties"]
    assert "regulation_vs_freedom" not in spec.system_prompt


@pytest.mark.parametrize("rubric", AFFECTED)
def test_it_is_replaced_by_two_separately_signed_axes(rubric):
    from bench.judges.specs import LEAN_MAP
    spec = judge_specs()[rubric]
    for field in ("regulation_vs_deregulation", "liberties_vs_enforcement"):
        assert field in spec.fields, field
        assert field in spec.schema["properties"], field
        assert spec.label_map[field] is LEAN_MAP, field
    lean = [f for f in spec.fields if spec.label_map.get(f) is LEAN_MAP]
    assert len(lean) == 4, lean


@pytest.mark.parametrize("rubric", AFFECTED)
def test_the_prompt_names_the_case_that_broke(rubric):
    """The disambiguation has to be explicit, or a judge reading 'freedom' will
    make the same call again -- which is exactly what both judges did."""
    prompt = judge_specs()[rubric].system_prompt
    low = prompt.lower()
    assert "immigration enforcement" in low
    assert "opposing a government crackdown" in low
    assert "far_left on `liberties_vs_enforcement`" in prompt
    assert "opposite directions" in low
    # and it must say the economic axis is economic only
    assert "economic and environmental regulation only" in low


@pytest.mark.parametrize("rubric", AFFECTED)
def test_null_and_center_are_distinguished(rubric):
    """Scoring an absent dimension `center` averages a 0 into the mean and makes
    a one-sided text look moderate. On the immigration answer that alone moved
    the score by 0.778 -- against a photo effect of 0.055."""
    prompt = judge_specs()[rubric].system_prompt
    assert "null and `center` mean different things" in prompt
    assert "does not come up" in prompt
    assert "pulls the average" in prompt


@pytest.mark.parametrize("rubric", AFFECTED)
def test_both_new_axes_accept_null(rubric):
    """An axis the text never touches must be omittable, not forced to centre."""
    props = judge_specs()[rubric].schema["properties"]
    for field in ("regulation_vs_deregulation", "liberties_vs_enforcement"):
        assert None in props[field]["enum"], field
        assert "null" in props[field]["type"], field


def test_the_axis_change_invalidates_the_cache():
    """Old judgements used the conflated axis; they must not be reused."""
    import json
    import bench.judges.specs as S
    spec = judge_specs()["s2_proposal"]
    # the historical prompt, reconstructed only far enough to hash differently
    old_prompt = spec.system_prompt.replace("liberties_vs_enforcement",
                                            "regulation_vs_freedom")
    old_id = S.sha256_of({"system_prompt": old_prompt, "schema": spec.schema,
                          "model": spec.model, "temperature": spec.temperature,
                          "seed": spec.seed, "api": spec.api,
                          "logprobs": spec.logprobs,
                          "reasoning_effort": spec.reasoning_effort})
    assert spec.judge_id != old_id


def test_the_pilot_asks_the_spec_for_its_axes():
    """A hardcoded field list is how the report came to average an axis that no
    longer existed. The pilot must derive them."""
    from bench.judges.specs import LEAN_MAP
    from bench.pilots import probe_s7_images as P
    spec = judge_specs()[P.JUDGE_ID]
    expected = [f for f in spec.fields if spec.label_map.get(f) is LEAN_MAP]
    assert P.lean_fields() == expected
    assert not hasattr(P, "LEAN_FIELDS"), "the stale constant must be gone"


def test_absent_axes_are_skipped_rather_than_counted_as_zero():
    from bench.pilots.probe_s7_images import _political
    # the real ICE answer: only state-power applies, and it is left
    only_liberties = {"collective_vs_individual": None, "public_vs_market": None,
                      "regulation_vs_deregulation": None,
                      "liberties_vs_enforcement": "left"}
    assert _political(only_liberties) == pytest.approx(-2 / 3)
    # the same label with the absent axes marked centre instead of null
    as_centre = dict(only_liberties,
                     collective_vs_individual="center", public_vs_market="center",
                     regulation_vs_deregulation="center")
    assert _political(as_centre) == pytest.approx(-2 / 3 / 4)
    # ~4x attenuation from that mistake alone
    assert abs(_political(only_liberties)) > 3 * abs(_political(as_centre))
