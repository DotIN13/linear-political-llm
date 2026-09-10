"""OpenAI-compatible judge caller (does NOT reuse the adaptor abstraction).

The adaptor is for the *subject* model; a judge is an offline pure function that
happens to be another LLM call. Mixing the two was the mistake the previous
design made, so this module is deliberately self-contained: it talks to the
endpoint directly via the ``openai`` client and returns schema-validated labels.

**Two APIs, because the model decides which one.** ``spec.api`` selects between
``/v1/chat/completions`` and ``/v1/responses``, and they are not the same call:
structured output moves from ``response_format={"type": "json_schema",
"json_schema": {...}}`` to ``text={"format": {"type": "json_schema", "name":
..., "strict": ..., "schema": ...}}``, ``messages`` becomes ``input``, and the
answer arrives on ``output_text`` rather than ``choices[0].message.content``.

Endpoint reality, measured against the live key on 2026-09-07 one parameter at
a time (the capability table lives below, next to the caller):

* ``gpt-5.4`` on chat completions takes strict json_schema **and** logprobs
  **and** ``temperature=0.0`` -- the only model on this key that takes all
  three, so it is the only bit-reproducible judge available. It is kept as the
  reproducibility check, not the default.
* ``gpt-5.6-luna`` and every other model newer than gpt-5.4 **refuse
  ``temperature``** ("Only the default (1) value is supported") and **refuse
  ``logprobs``** ("not supported with this model"). On the Responses API
  ``seed`` and ``top_p`` are refused as well, and ``reasoning.effort`` and
  ``max_output_tokens`` appear instead. luna is the default judge; see the
  ``DEFAULT_JUDGE_MODEL`` comment for why and what it costs.
* DeepSeek (``DEEPSEEK_API_KEY``) does **not** support strict json_schema (400
  "This response_format type is unavailable now"); it has JSON mode
  (``json_object``) and logprobs.

A refused parameter is a hard 400 that aborts the call, not a warning -- so
every optional parameter is omitted rather than sent as ``None``, and what to
omit comes from the spec instead of being guessed here.

The primary path is strict json_schema; when the endpoint refuses it the caller
falls back to JSON mode plus client-side schema validation and one retry.
logprobs is requested and stored where available but is never part of the
primary scale (the seven-point label is the scale), so a model without them
loses nothing measured.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from bench_v2.judge.schema import strict_schema
from bench_v2.types import sha256_of

# gpt-5.6-luna with reasoning off, the team's default as of 2026-09-10. It is not
# bit-reproducible the way gpt-5.4 was (no temperature, no seed, and the Responses
# API), but a side-by-side on the 1188 s1_speech v3 answers showed its labels agree
# with gpt-5.4 on 76% of `lean` (Pearson +0.67) and the memory-minus-bare effect it
# reports is the same size and sign. It reads ~0.06 further left overall, which
# matters for *levels*, not for the differences the hypothesis turns on.
#
# gpt-5.4 remains reachable for a reproducibility check with
# BENCH_JUDGE_MODEL=gpt-5.4; that is also the only model here that accepts strict
# json_schema **and** logprobs **and** temperature=0.0 at once.
DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"
DEFAULT_SEED = 20260905


@dataclass(frozen=True)
class ModelCaps:
    """What one judge model will actually accept, measured against the live key.

    Every model newer than gpt-5.4 rejects ``temperature`` and ``logprobs``
    outright; on the Responses API ``seed`` is not a parameter at all and
    ``reasoning.effort`` appears instead. A refused parameter is a hard 400, so
    the caller omits whatever this table says None.
    """

    api: str                                # "chat" | "responses"
    temperature: float | None
    seed: int | None
    logprobs: bool
    reasoning_effort: str | None = None


MODEL_CAPS: dict[str, ModelCaps] = {
    "gpt-5.4": ModelCaps(api="chat", temperature=0.0, seed=DEFAULT_SEED,
                         logprobs=True),
    # Default judge. Reasoning off: on a 1188-row run effort="high" is minutes per
    # task instead of seconds, and the labels move no more than between two
    # reasoning-on calls. ``BENCH_JUDGE_REASONING_EFFORT`` raises it if wanted.
    "gpt-5.6-luna": ModelCaps(api="responses", temperature=None, seed=None,
                              logprobs=False, reasoning_effort="none"),
    "gpt-5.6-sol": ModelCaps(api="responses", temperature=None, seed=None,
                             logprobs=False, reasoning_effort="high"),
    "gpt-5.6-terra": ModelCaps(api="responses", temperature=None, seed=None,
                               logprobs=False, reasoning_effort="high"),
    "gpt-5.5": ModelCaps(api="responses", temperature=None, seed=None,
                         logprobs=False, reasoning_effort="high"),
    "gpt-6-astra": ModelCaps(api="responses", temperature=None, seed=None,
                             logprobs=False, reasoning_effort="high"),
}

UNKNOWN_MODEL_CAPS = ModelCaps(api="responses", temperature=None, seed=None,
                               logprobs=False, reasoning_effort="high")


def caps_for(model: str) -> ModelCaps:
    """The model's measured capabilities, with one env override.

    ``BENCH_JUDGE_REASONING_EFFORT`` overrides a Responses model's
    ``reasoning.effort`` for a run. The default is ``none`` (off), which is what
    makes the default judge fast enough for a 1000-row run; the override raises
    it (``low``/``minimal``/``high``) when a check needs it. It only ever touches
    ``reasoning_effort`` -- temperature, seed and logprobs stay as measured --
    and because effort enters ``judge_id``, an overridden run is a distinct
    cache key rather than a collision with the default one.
    """
    caps = MODEL_CAPS.get(model, UNKNOWN_MODEL_CAPS)
    override = os.environ.get("BENCH_JUDGE_REASONING_EFFORT")
    if override:
        caps = ModelCaps(api=caps.api, temperature=caps.temperature,
                         seed=caps.seed, logprobs=caps.logprobs,
                         reasoning_effort=override)
    return caps


@dataclass(frozen=True)
class JudgeSpec:
    """One judge: model + criteria prompt + a Pydantic response model + label map.

    The *data* is task-specific and lives in the task's ``judge_spec.py`` -- the
    criteria prompt, the ``response_model`` subclass of ``JudgeLabels``, and the
    label->value map. This class is the shape the caller needs, plus the cache key
    that must be computed the same way for every task.

    ``judge_id = sha256(system_prompt + schema + model + temperature + seed)``,
    where ``schema`` is the strict JSON schema derived from ``response_model``.
    Only non-default optional fields enter the hash, so a spec that still uses
    chat completions with logprobs and no reasoning effort hashes exactly as it
    did before those fields existed, and the gpt-5.4 cache rows stay reachable.
    """

    id: str
    model: str
    system_prompt: str
    response_model: Any                      # a pydantic BaseModel subclass
    label_map: dict[str, dict[str, float]]
    fields: list[str]
    temperature: float | None = 0.0
    seed: int | None = DEFAULT_SEED
    logprobs: bool = True
    api: str = "chat"
    reasoning_effort: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"

    @property
    def schema(self) -> dict[str, Any]:
        """The strict JSON schema the endpoint is given (and the cache hashes)."""
        return strict_schema(self.response_model)

    @property
    def judge_id(self) -> str:
        payload: dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "schema": self.schema,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if self.api != "chat":
            payload["api"] = self.api
        if not self.logprobs:
            payload["logprobs"] = False
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return sha256_of(payload)

    @property
    def label_fields(self) -> list[str]:
        return ["rationale", "political_content_present", "refusal", *self.fields]


class JudgeError(RuntimeError):
    """A judge call could not be completed even after fallbacks."""


def _schema_to_response_format(spec: JudgeSpec) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": spec.id, "strict": True, "schema": spec.schema},
    }


def _openai_client(spec: JudgeSpec):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - env dependent
        raise JudgeError("openai package not installed; cannot run the judge") from exc
    key = os.environ.get(spec.api_key_env)
    if not key:
        raise JudgeError(f"no API key in ${spec.api_key_env}; judge cannot run")
    kwargs: dict[str, Any] = {"api_key": key}
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    return OpenAI(**kwargs)


def _messages(spec: JudgeSpec, text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": text},
    ]


def _request(
    client: Any,
    spec: JudgeSpec,
    text: str,
    response_format: dict[str, Any],
    logprobs: bool,
) -> Any:
    """One judge call, on whichever API the spec names.

    **Every optional parameter is omitted when the spec says None**, because the
    newer models 400 on a parameter they do not support rather than ignoring it
    -- `temperature` is refused outright by everything past gpt-5.4, and `seed`
    is not a Responses parameter at all. Sending them "just in case" fails the
    whole call, so the capability table in specs.py decides and this function
    only obeys it.
    """
    if spec.api == "responses":
        return _responses_request(client, spec, text, response_format)

    kwargs: dict[str, Any] = {
        "model": spec.model,
        "response_format": response_format,
        "messages": _messages(spec, text),
    }
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if spec.seed is not None:
        kwargs["seed"] = spec.seed
    if logprobs and spec.logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    return client.chat.completions.create(**kwargs)


def _responses_request(
    client: Any,
    spec: JudgeSpec,
    text: str,
    response_format: dict[str, Any],
) -> Any:
    """The /v1/responses shape, which is not the chat-completions shape.

    Structured output moves from `response_format={"type": "json_schema",
    "json_schema": {...}}` to `text={"format": {"type": "json_schema", "name":
    ..., "strict": ..., "schema": ...}}` -- the schema fields are hoisted one
    level up, and `name` sits beside `schema` rather than wrapping it. Verified
    against the live endpoint before being written here.
    """
    fmt = _to_responses_format(response_format, spec.id)
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "input": _messages(spec, text),
        "text": {"format": fmt},
    }
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if spec.reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": spec.reasoning_effort}
    return client.responses.create(**kwargs)


def _to_responses_format(response_format: dict[str, Any], name: str) -> dict[str, Any]:
    """Translate a chat-completions response_format into the Responses shape."""
    if response_format.get("type") == "json_schema":
        inner = response_format.get("json_schema") or {}
        return {
            "type": "json_schema",
            "name": inner.get("name") or name,
            "strict": inner.get("strict", True),
            "schema": inner.get("schema") or {},
        }
    # JSON mode is spelled the same way in both APIs.
    return {"type": "json_object"}


def _validated(spec: JudgeSpec, payload: Any) -> dict[str, Any] | None:
    """The payload as validated labels, or None when the model rejects it.

    The task's Pydantic model does the checking -- required fields, enums,
    ``extra="forbid"`` -- so there is no second schema implementation here.
    """
    if payload is None:
        return None
    try:
        return spec.response_model.model_validate(payload).model_dump(mode="json")
    except ValidationError:
        return None


def _parse_content(choice: Any) -> str:
    content = getattr(getattr(choice, "message", None), "content", None)
    if content is None:
        return ""
    if isinstance(content, list):           # some providers return a list of parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _response_text(resp: Any) -> str:
    """The judge's JSON, from either API.

    Chat completions puts it on `choices[0].message.content`; the Responses API
    exposes `output_text`, with the structured list on `output` as a fallback --
    and on a reasoning model that list also holds reasoning items, which carry
    no `text` and must be skipped rather than concatenated.
    """
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    output = getattr(resp, "output", None)
    if output:
        parts: list[str] = []
        for item in output:
            for piece in (getattr(item, "content", None) or []):
                piece_text = getattr(piece, "text", None)
                if isinstance(piece_text, str):
                    parts.append(piece_text)
        if parts:
            return "".join(parts)
    choices = getattr(resp, "choices", None)
    if choices:
        return _parse_content(choices[0])
    return ""


class JudgeCaller:
    """One stateless, single-turn call per answer (board step 3)."""

    def __init__(self, spec: JudgeSpec) -> None:
        self.spec = spec

    def call(self, text: str) -> dict[str, Any]:
        """Judge one answer. Returns the validated label payload plus provenance.

        Raises JudgeError only after strict-schema, JSON-mode-with-validation and
        the logprobs fallbacks have all been tried.
        """
        client = _openai_client(self.spec)
        started = time.time()

        try:
            resp = _request(client, self.spec, text,
                            _schema_to_response_format(self.spec), logprobs=True)
        except Exception as exc:  # noqa: BLE001 - the fallback chain decides what to do
            resp = self._fallback(client, text, exc)

        labels = _validated(self.spec, _loads(_response_text(resp)))
        if labels is None:
            # The model produced something the endpoint accepted but the Pydantic
            # model rejects; one retry in JSON mode with the schema echoed.
            resp = self._json_mode_call(client, text)
            labels = _validated(self.spec, _loads(_response_text(resp)))
            if labels is None:
                raise JudgeError("judge output failed pydantic validation after retry")

        logprobs = _extract_logprobs(resp)
        return {
            "judge_id": self.spec.judge_id,
            "model": self.spec.model,
            "labels": labels,
            "logprobs": logprobs,
            "timing_ms": (time.time() - started) * 1000.0,
            "usage": getattr(resp, "usage", None).model_dump()
            if hasattr(getattr(resp, "usage", None), "model_dump") else None,
        }

    def _fallback(self, client: Any, text: str, first_error: Exception) -> Any:
        """Endpoint refused the strict-schema request -> JSON mode."""
        return self._json_mode_call(client, text)

    def _json_mode_call(self, client: Any, text: str) -> Any:
        # Try with logprobs on, then off. The spec already knows whether the
        # model supports them; the second attempt covers an endpoint that
        # accepts the parameter and then refuses this particular request.
        try:
            return _request(client, self.spec, text, {"type": "json_object"}, logprobs=True)
        except Exception:  # noqa: BLE001
            return _request(client, self.spec, text, {"type": "json_object"}, logprobs=False)


def _loads(content: str) -> Any | None:
    content = (content or "").strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # strip a ```json fence if the model wrapped it
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
        return None


def _extract_logprobs(resp: Any) -> list[Any] | None:
    """Store token logprobs as a fallback diagnostic; never part of the scale."""
    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return None
    content = getattr(logprobs, "content", None)
    if content is None:
        return None
    out = []
    for token in content:
        out.append({
            "token": getattr(token, "token", None),
            "logprob": getattr(token, "logprob", None),
            "top": [(t.token, t.logprob)
                    for t in (getattr(token, "top_logprobs", None) or [])],
        })
    return out
