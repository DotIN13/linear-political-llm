"""Adaptor protocol and the capability gate.

The gate is the whole point of the design (docs/bench/01): a surface says what
it needs, an adaptor says what it can produce, and ``bench check`` refuses or
warns *before* you spend hours in a queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from bench.types import Capability, Response, Trial


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
