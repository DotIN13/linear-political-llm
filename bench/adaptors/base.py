"""Adaptor protocol and the capability gate.

The gate is the whole point of the design (docs/bench/01): a surface says what
it needs, an adaptor says what it can produce, and ``bench check`` refuses or
warns *before* you spend hours in a queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from bench.types import Capability, Item, Response, Trial, baseline_item


@runtime_checkable
class Adaptor(Protocol):
    name: str
    capabilities: frozenset

    def run(self, trial: Trial) -> Response: ...


class BaseAdaptor:
    """Small shared base. Adaptors are constructed by the CLI, not discovered."""

    name: str = "base"
    capabilities: frozenset = frozenset()

    def __init__(self, model: str = "", seed: int = 42, **kwargs: Any) -> None:
        self.model = model
        self.seed = seed
        self.options: Dict[str, Any] = dict(kwargs)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": str(self.model),
            "seed": self.seed,
            "capabilities": sorted(str(c) for c in self.capabilities),
            "options": {k: v for k, v in self.options.items() if _jsonable(v)},
        }

    def setup(self) -> None:
        """Load weights / open connections. Called once before the first trial."""

    def teardown(self) -> None:
        """Release resources."""

    def run(self, trial: Trial) -> Response:
        raise NotImplementedError


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


@dataclass
class GateReport:
    surface: str
    adaptor: str
    ok: bool                                    # can it run at all?
    degraded: bool                              # runs, but produces less
    missing_required: List[str] = field(default_factory=list)
    degradations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.ok:
            return "BLOCKED"
        return "DEGRADED" if self.degraded else "OK"

    def render(self) -> str:
        lines = [f"{self.surface} x {self.adaptor}: {self.status}"]
        req_mark = "x" if self.missing_required else "OK"
        lines.append(f"  [{req_mark}] requires satisfied")
        for miss in self.missing_required:
            lines.append(f"      missing required capability: {miss}")
        for note in self.notes:
            lines.append(f"  [OK] {note}")
        for deg in self.degradations:
            lines.append(f"  [!]  {deg}")
        if self.ok and not self.degraded:
            lines.append("  -> runs at full fidelity")
        elif self.ok:
            lines.append("  -> runs DEGRADED (see above); records will carry nulls for the missing fields")
        else:
            lines.append("  -> refusing to run this combination")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface": self.surface,
            "adaptor": self.adaptor,
            "status": self.status,
            "ok": self.ok,
            "degraded": self.degraded,
            "missing_required": self.missing_required,
            "degradations": self.degradations,
            "notes": self.notes,
        }


def check_capabilities(surface: Any, adaptor: Any, item: Optional[Any] = None) -> GateReport:
    """Compare surface.requires / surface.prefers against adaptor.capabilities.

    ``requires`` is the hard minimum: without it the surface cannot produce any
    outcome and the gate blocks. ``prefers`` is what the surface would use to
    produce its *primary* measurement; missing those means degraded, not blocked
    -- which is exactly the vote2020 x opencode case (no logprob, no s_txt, so
    the outcome falls back to parsing generated text).
    """

    caps = frozenset(adaptor.capabilities)
    required = frozenset(getattr(surface, "requires", frozenset()))
    preferred = frozenset(getattr(surface, "prefers", frozenset()))

    missing_required = sorted(str(c) for c in (required - caps))
    notes: List[str] = []
    degradations: List[str] = []

    if not missing_required:
        notes.append(f"requires {{{', '.join(sorted(str(c) for c in required))}}} -> all present")

    if Capability.LOGPROB in preferred and Capability.LOGPROB not in caps:
        degradations.append(
            "surface scores candidate words by logprob but the adaptor has no `logprob`: "
            "outcome falls back to parsing the generated text (extract()), "
            "which reintroduces the ceiling/parse-failure problems this surface exists to avoid"
        )

    probe_points = []
    if hasattr(surface, "probe_points"):
        try:
            probe_points = list(surface.probe_points(None))
        except Exception:
            probe_points = []
    if probe_points and Capability.ACTIVATIONS not in caps:
        names = ", ".join(p.name for p in probe_points)
        degradations.append(
            f"probe_points is non-empty ({names}) but the adaptor has no `activations`: "
            f"the surface runs degraded and produces no {names}"
        )
    elif probe_points:
        notes.append(f"probe_points {{{', '.join(p.name for p in probe_points)}}} readable via `activations`")

    if getattr(surface, "uses_images", True) and Capability.IMAGES not in caps:
        missing_required.append(str(Capability.IMAGES))

    return GateReport(
        surface=getattr(surface, "name", str(surface)),
        adaptor=getattr(adaptor, "name", str(adaptor)),
        ok=not missing_required,
        degraded=bool(degradations),
        missing_required=sorted(set(missing_required)),
        degradations=degradations,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# The candidate gate (task A3). Lives next to the capability gate because it is
# the same kind of thing: a cheap pre-flight check that refuses a combination
# *before* the queue, not after.
# --------------------------------------------------------------------------- #
@dataclass
class CandidateReport:
    surface: str
    adaptor: str
    condition: str
    variant: Dict[str, Any]
    single_token: Dict[str, bool] = field(default_factory=dict)
    token_ids: Dict[str, List[int]] = field(default_factory=dict)
    argmax_token: Optional[str] = None
    argmax_hits: Optional[bool] = None
    top_tokens: List[Any] = field(default_factory=list)
    logprobs: Dict[str, float] = field(default_factory=dict)
    question: str = ""
    error: Optional[str] = None
    applicable: bool = True              # False when the surface declares no candidates

    @property
    def ok(self) -> bool:
        if not self.applicable:
            return True                  # no candidates to gate
        return (self.error is None
                and bool(self.single_token) and all(self.single_token.values())
                and self.argmax_hits is True)

    def render(self) -> str:
        lines = [f"{self.surface} x {self.adaptor}: candidates {'OK' if self.ok else 'FAILED'}"
                 f"  (condition={self.condition}, variant={self.variant})"]
        for candidate, single in sorted(self.single_token.items()):
            ids = self.token_ids.get(candidate, [])
            mark = "OK" if single else "x "
            lines.append(f"  [{mark}] {candidate!r} -> token ids {ids} "
                         f"({len(ids)} token{'' if len(ids) == 1 else 's'})")
        if self.error:
            lines.append(f"  [x ] forward pass failed: {self.error}")
            return "\n".join(lines)
        if self.argmax_hits is None:
            lines.append("  [--] argmax not checked (adaptor cannot return logprobs)")
            return "\n".join(lines)
        mark = "OK" if self.argmax_hits else "x "
        lines.append(f"  [{mark}] argmax at the answer position = {self.argmax_token!r} "
                     f"({'in' if self.argmax_hits else 'NOT in'} {sorted(self.single_token)})")
        if self.logprobs:
            pretty = "  ".join(f"logP({k})={v:+.4f}" for k, v in sorted(self.logprobs.items()))
            lines.append(f"       {pretty}")
        if self.top_tokens:
            pretty = ", ".join(f"{tok!r}:{lp:+.3f}" for tok, lp in self.top_tokens)
            lines.append(f"       top-5 at that position: {pretty}")
        if not self.argmax_hits:
            lines.append("  -> the prompt does not put the model in single-letter mode; "
                         "this surface is unusable as measured")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface": self.surface, "adaptor": self.adaptor, "condition": self.condition,
            "variant": self.variant, "ok": self.ok, "single_token": self.single_token,
            "token_ids": self.token_ids, "argmax_token": self.argmax_token,
            "argmax_hits": self.argmax_hits, "top_tokens": self.top_tokens,
            "logprobs": self.logprobs, "question": self.question, "error": self.error,
        }


def check_candidates(
    surface: Any,
    adaptor: Any,
    item: Optional[Item] = None,
    condition: str = "E",
    variant: Optional[Dict[str, Any]] = None,
) -> CandidateReport:
    """Are the candidate tokens single tokens, and does the model actually answer with one?

    Two failures this catches, both of which silently produce numbers that look
    fine: a candidate that tokenizes to more than one token (then the logprob is
    a first-sub-token logprob, not the option's), and a prompt the model does not
    read as "reply with a letter" (then the difference of two letter logprobs is
    measured off in the tail of the distribution).
    """
    variant = dict(variant or {"phrasing": 0, "order": "ab"})
    item = item or baseline_item()
    trial = surface.build(item, condition, variant)
    report = CandidateReport(
        surface=getattr(surface, "name", str(surface)),
        adaptor=getattr(adaptor, "name", str(adaptor)),
        condition=condition,
        variant=variant,
        question=trial.meta.get("question", ""),
    )
    if not trial.candidates:
        report.applicable = False        # generation surfaces: nothing to gate
        return report

    tokenize = getattr(adaptor, "tokenize", None)
    if callable(tokenize):
        for candidate in trial.candidates:
            ids = list(tokenize(candidate))
            report.token_ids[candidate] = ids
            report.single_token[candidate] = len(ids) == 1
    else:
        report.error = "adaptor exposes no tokenizer"
        return report

    if Capability.LOGPROB not in frozenset(adaptor.capabilities):
        return report                      # argmax_hits stays None: not checkable here

    try:
        response = adaptor.run(trial)
    except Exception as exc:               # a failed pre-flight must not look like a pass
        report.error = f"{type(exc).__name__}: {exc}"
        return report
    if response.error:
        report.error = response.error
        return report

    report.logprobs = dict(response.logprobs or {})
    report.argmax_token = response.usage.get("argmax_token")
    report.top_tokens = list(response.usage.get("top_tokens") or [])
    report.argmax_hits = report.argmax_token in trial.candidates
    return report
