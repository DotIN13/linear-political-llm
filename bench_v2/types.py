"""Core value types for the bench harness.

Five abstractions (docs/bench/01): Item, Surface, Adaptor, Judge, Store.
This module holds the data that flows between them, plus the capability enum
that the gate in ``bench.adaptors.base`` checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class Capability(str, Enum):
    """What a backend can produce. Surfaces declare what they need."""

    GENERATE = "generate"
    LOGPROB = "logprob"
    ACTIVATIONS = "activations"
    STEER = "steer"
    IMAGES = "images"
    SESSION = "session"

    def __str__(self) -> str:  # keeps f-strings and json readable
        return self.value


ALL_CAPABILITIES = frozenset(Capability)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_variant(variant: Optional[Dict[str, Any]]) -> str:
    """Canonical form of the within-item repeat dimensions (docs/bench, task B).

    A dict rather than two fields so that later repeats -- image position, number
    of images -- can be added without touching the signature again. The empty
    dict has the determinate form ``{}``.
    """
    return _canonical_json(dict(variant or {}))


@dataclass(frozen=True)
class Item:
    """One stimulus. Produced by the sampler, then frozen on disk forever."""

    item_id: str
    images: List[str]              # record_name, e.g. "train2017/000000000030.jpg"
    image_paths: List[str]         # resolved on-disk paths (already resized to 800px)
    image_scores: List[float]      # per-image probe image_mean
    stratum: int                   # 0..9 over image_mean -- THE primary independent variable
    primary_iv: str = "stratum"    # explicit, so nobody quietly regresses on image_mean_mean
    covariates: Dict[str, Any] = field(default_factory=dict)
    split: str = "explore"

    @property
    def image_mean(self) -> Optional[float]:
        """Mean of the per-image scores. A *covariate* now, not the IV (task E)."""
        return sum(self.image_scores) / len(self.image_scores) if self.image_scores else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Item":
        """**The one way an Item is built from disk**, and the only place that
        decides where its images are.

        ``image_paths`` on disk are whatever the machine that sampled the set
        wrote -- absolute, and therefore wrong on every other machine and in
        every worktree that does not own a copy of the images. ``images`` holds
        record names, which are stable.

        So the resolved paths are **recomputed from the record names** rather
        than read. With ``LPL_IMAGES_ROOT`` unset this reproduces the old layout
        exactly, so nothing changes for an existing checkout; set it, and one
        items file is correct everywhere.

        A set with no ``images`` (the baseline item, and a few older files) keeps
        whatever ``image_paths`` it had -- there is nothing to resolve from.
        """
        from bench_v2.paths import resolve_image

        payload = dict(payload)
        if "stratum" not in payload and "decile" in payload:
            payload["stratum"] = payload["decile"]      # v1 items on disk say "decile"
        records = payload.get("images") or []
        if records:
            frozen = list(payload.get("image_paths") or [])
            frozen += [None] * (len(records) - len(frozen))
            payload["image_paths"] = [resolve_image(r, f)
                                      for r, f in zip(records, frozen)]
        known = {f for f in Item.__dataclass_fields__}
        return Item(**{k: v for k, v in payload.items() if k in known})


# The one record written for an item-invariant condition (task C). Its conversation
# carries no image, so every item would otherwise produce a byte-identical trial.
BASELINE_ITEM_ID = "__baseline__"


def baseline_item(split: str = "explore") -> Item:
    return Item(item_id=BASELINE_ITEM_ID, images=[], image_paths=[], image_scores=[],
                stratum=-1, covariates={}, split=split)


@dataclass(frozen=True)
class Conversation:
    """A fully built chat, in the message shape token_scoring.py already accepts.

    ``messages`` is a list of ``{"role", "content"}`` where content is a list of
    ``{"type": "text"|"image", ...}`` parts. Every assistant turn present here is
    a hard-coded constant -- the model never generates it (docs/bench/03).
    """

    messages: List[Dict[str, Any]]
    images: List[str] = field(default_factory=list)

    @property
    def sha(self) -> str:
        return sha256_of({"messages": self.messages, "images": self.images})

    def to_dict(self) -> Dict[str, Any]:
        return {"messages": self.messages, "images": self.images, "sha": self.sha}

    def render_text(self) -> str:
        """Flatten to plain text for backends that cannot take structured turns."""
        lines = []
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part["text"])
                    elif part.get("type") == "image":
                        parts.append("[image]")
                text = " ".join(parts)
            else:
                text = str(content)
            lines.append(f"{message['role'].upper()}: {text}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ProbePoint:
    """Where to read the probe during the forward pass."""

    name: str                      # "s_txt" / "s_img"
    kind: str                      # "last_text" | "image_tokens"
    reduce: str = "mean"           # "last" | "mean"


@dataclass(frozen=True)
class Trial:
    """One unit of work: a conversation plus what to measure on it."""

    surface: str
    item_id: str
    condition: str
    conversation: Conversation
    candidates: List[str] = field(default_factory=list)   # tokens to take logprob of ("A"/"B")
    probe_points: List[ProbePoint] = field(default_factory=list)
    max_new_tokens: int = 0                               # 0 == prefill only
    variant: Dict[str, Any] = field(default_factory=dict)  # {"phrasing": 0, "order": "ab"}
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def variant_key(self) -> str:
        return canonical_variant(self.variant)


@dataclass
class Response:
    """What an adaptor gives back. Fields an adaptor cannot fill stay None."""

    text: Optional[str] = None
    logprobs: Optional[Dict[str, float]] = None
    probe: Optional[Dict[str, Any]] = None
    session_log: Optional[List[Dict[str, Any]]] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    timing_ms: Optional[float] = None
    cost_usd: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "logprobs": self.logprobs,
            "session_log": self.session_log,
            "usage": self.usage,
        }


@dataclass(frozen=True)
class Outcome:
    """The dependent variable, extracted deterministically where possible."""

    kind: str                      # e.g. "logprob_diff", "choice_text"
    value: Optional[float]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "value": self.value, "extra": self.extra}


@dataclass(frozen=True)
class NeedsJudge:
    """Returned by ``Surface.extract`` when no deterministic parse exists."""

    reason: str
    judge_hint: Optional[str] = None


def capabilities_from_names(names: Sequence[str]) -> frozenset:
    return frozenset(Capability(str(n)) for n in names)
