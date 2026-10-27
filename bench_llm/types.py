"""The values that flow through a run: Item -> Trial -> Response -> Outcome."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Item:
    """One stimulus: an id, its image files, and whatever the pilot wants to carry."""

    item_id: str
    image_paths: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Conversation:
    """Chat messages; content is a list of ``{"type": "text"|"image", ...}`` parts."""

    messages: List[Dict[str, Any]]

    @property
    def sha(self) -> str:
        return sha256_of(self.messages)


@dataclass(frozen=True)
class Trial:
    item_id: str
    conversation: Conversation
    variant: Dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 16
    top_logprobs: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """What an adaptor returns. ``logprobs`` is the first generated token's top tokens;
    ``probe`` is what an adaptor that can read activations found in them (None otherwise)."""

    text: Optional[str] = None
    logprobs: Optional[Dict[str, float]] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    timing_ms: Optional[float] = None
    error: Optional[str] = None
    probe: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"text": self.text, "logprobs": self.logprobs, "usage": self.usage,
               "timing_ms": self.timing_ms, "error": self.error}
        if self.probe is not None:
            out["probe"] = self.probe
        return out


@dataclass(frozen=True)
class Outcome:
    """The dependent variable: a number or None, and whatever was read on the way."""

    value: Optional[float]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "extra": self.extra}
