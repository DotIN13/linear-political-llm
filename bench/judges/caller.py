"""OpenAI-compatible judge caller (does NOT reuse the adaptor abstraction).

The adaptor is for the *subject* model; a judge is an offline pure function that
happens to be another LLM call. Mixing the two was the mistake the previous
design made, so this module is deliberately self-contained: it talks to
``{base_url}/v1/chat/completions`` directly via the ``openai`` client and returns
schema-validated labels.

Endpoint reality (checked on the login node, recorded in the report):

* OpenAI (``OPENAI_API_KEY``) supports ``response_format: json_schema`` with
  ``strict: true`` **and** ``logprobs``/``top_logprobs``.
* DeepSeek (``DEEPSEEK_API_KEY``) does **not** support strict json_schema (400
  "This response_format type is unavailable now"); it has JSON mode
  (``json_object``) and logprobs.

So the primary path is strict json_schema; when the endpoint refuses it the
caller falls back to JSON mode plus client-side schema validation and one retry.
logprobs is requested and stored when the endpoint honours it, but it is never
part of the primary scale (board: the seven-point label is the scale).
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


def _request(
    client: Any,
    spec: JudgeSpec,
    text: str,
    response_format: Dict[str, Any],
    logprobs: bool,
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": spec.model,
        "temperature": spec.temperature,
        "seed": spec.seed,
        "response_format": response_format,
        "messages": [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": text},
        ],
    }
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    return client.chat.completions.create(**kwargs)


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

        content = _parse_content(resp.choices[0])
        payload = _loads(content)

        problem = _validate(payload, self.spec.schema) if payload is not None else "unparseable"
        if problem:
            # The model produced something out of schema even though the endpoint
            # accepted the request; one retry in JSON mode with the schema echoed.
            resp = self._json_mode_call(client, text)
            content = _parse_content(resp.choices[0])
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
        # Try with logprobs on, then off.
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
