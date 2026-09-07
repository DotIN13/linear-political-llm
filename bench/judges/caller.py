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
a time (the capability table lives in ``specs.py``):

* ``gpt-5.4`` on chat completions takes strict json_schema **and** logprobs
  **and** ``temperature=0.0`` -- the only model on this key that takes all
  three, so it is the only bit-reproducible judge available.
* ``gpt-5.6-luna`` and every other model newer than gpt-5.4 **refuse
  ``temperature``** ("Only the default (1) value is supported") and **refuse
  ``logprobs``** ("not supported with this model"). On the Responses API
  ``seed`` and ``top_p`` are refused as well, and ``reasoning.effort`` and
  ``max_output_tokens`` appear instead.
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
from typing import Any, Dict, List, Optional, Tuple

from bench.judges.specs import JudgeSpec


class JudgeError(RuntimeError):
    """A judge call could not be completed even after fallbacks."""


def _schema_to_response_format(spec: JudgeSpec) -> Dict[str, Any]:
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
    kwargs: Dict[str, Any] = {"api_key": key}
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    return OpenAI(**kwargs)


def _messages(spec: JudgeSpec, text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": text},
    ]


def _request(
    client: Any,
    spec: JudgeSpec,
    text: str,
    response_format: Dict[str, Any],
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

    kwargs: Dict[str, Any] = {
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
    response_format: Dict[str, Any],
) -> Any:
    """The /v1/responses shape, which is not the chat-completions shape.

    Structured output moves from `response_format={"type": "json_schema",
    "json_schema": {...}}` to `text={"format": {"type": "json_schema", "name":
    ..., "strict": ..., "schema": ...}}` -- the schema fields are hoisted one
    level up, and `name` sits beside `schema` rather than wrapping it. Verified
    against the live endpoint before being written here.
    """
    fmt = _to_responses_format(response_format, spec.id)
    kwargs: Dict[str, Any] = {
        "model": spec.model,
        "input": _messages(spec, text),
        "text": {"format": fmt},
    }
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if spec.reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": spec.reasoning_effort}
    return client.responses.create(**kwargs)


def _to_responses_format(response_format: Dict[str, Any], name: str) -> Dict[str, Any]:
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


def _validate(payload: Any, schema: Dict[str, Any]) -> Optional[str]:
    """Return a problem string, or None when the payload satisfies the schema.

    Deliberately small: it checks the things that actually matter for downstream
    analysis -- it is an object, has the required keys, and enum/boolean/string
    fields carry legal values. It does not try to reimplement JSON Schema.
    """
    if not isinstance(payload, dict):
        return "output is not a JSON object"
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in payload:
            return f"missing required field {key!r}"
    for key, prop in props.items():
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(prop.get("type"), list):       # nullable
            if value is None:
                continue
            base = [t for t in prop["type"] if t != "null"]
            if base and not isinstance(value, _py_type(base[0])):
                return f"field {key!r} has wrong type"
        elif prop.get("type") == "boolean":
            if not isinstance(value, bool):
                return f"field {key!r} is not a boolean"
        elif prop.get("type") == "array":
            if not isinstance(value, list):
                return f"field {key!r} is not an array"
        else:
            if not isinstance(value, str):
                return f"field {key!r} is not a string"
        enum = prop.get("enum")
        if enum is not None and value is not None and value not in enum:
            return f"field {key!r}={value!r} not in enum"
    if schema.get("additionalProperties") is False:
        extra = set(payload) - set(props)
        if extra:
            return f"unexpected fields {sorted(extra)}"
    return None


def _py_type(name: str) -> type:
    return {"string": str, "number": (int, float), "integer": int, "boolean": bool,
            "array": list, "object": dict}.get(name, object)


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
        parts: List[str] = []
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

    def call(self, text: str) -> Dict[str, Any]:
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

        content = _response_text(resp)
        payload = _loads(content)

        problem = _validate(payload, self.spec.schema) if payload is not None else "unparseable"
        if problem:
            # The model produced something out of schema even though the endpoint
            # accepted the request; one retry in JSON mode with the schema echoed.
            resp = self._json_mode_call(client, text)
            content = _response_text(resp)
            payload = _loads(content)
            problem = _validate(payload, self.spec.schema) if payload is not None else "unparseable"
            if problem:
                raise JudgeError(f"judge output failed schema validation after retry: {problem}")

        logprobs = _extract_logprobs(resp)
        return {
            "judge_id": self.spec.judge_id,
            "model": self.spec.model,
            "labels": payload,
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


def _loads(content: str) -> Optional[Any]:
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


def _extract_logprobs(resp: Any) -> Optional[List[Any]]:
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
