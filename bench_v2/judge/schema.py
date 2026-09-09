"""Pydantic models for judge output, and the strict JSON schema the endpoint wants.

Every judge returns the same three bookkeeping fields -- ``rationale``,
``political_content_present``, ``refusal`` -- plus its task-specific labels. The
base model holds the three; each task's ``judge_spec.py`` subclasses it and adds
its own fields. ``extra="forbid"`` makes the schema
``additionalProperties: false`` and turns an out-of-schema answer into a
validation error rather than a silently ignored key.

``strict_schema`` renders a model into the shape the endpoint accepts:

* ``$ref`` inlined from ``$defs`` (strict mode does not want references);
* ``title`` and ``default`` stripped;
* a nullable enum as ``{"type": ["string", "null"], "enum": [..., null]}``;
* every property listed in ``required``.

It is deliberately deterministic, because the schema is part of ``judge_id``: a
change here is a new judge, not a cache hit.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class JudgeLabels(BaseModel):
    """The fields every judge returns, before the task's own labels."""

    model_config = ConfigDict(extra="forbid")

    rationale: str
    political_content_present: bool
    refusal: bool


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return resolve(defs[node["$ref"].split("/")[-1]])
        if "anyOf" in node:
            branches = node["anyOf"]
            non_null = [b for b in branches if b.get("type") != "null"]
            has_null = any(b.get("type") == "null" for b in branches)
            if len(non_null) == 1 and has_null:
                merged = resolve(non_null[0])
                if "enum" in merged:
                    return {"type": ["string", "null"], "enum": list(merged["enum"]) + [None]}
                return {**merged, "type": [merged.get("type", "string"), "null"]}
            return {"anyOf": [resolve(b) for b in branches]}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in {"title", "default", "$defs"}:
                continue
            if key == "properties":
                out[key] = {name: resolve(prop) for name, prop in value.items()}
            else:
                out[key] = resolve(value) if isinstance(value, (dict, list)) else value
        return out

    schema = resolve(raw)
    schema["required"] = list(schema.get("properties", {}))
    return schema
