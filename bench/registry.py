"""Three decorators. Not a plugin system -- imports are explicit (docs/bench/02)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

_SURFACES: Dict[str, Any] = {}
_ADAPTORS: Dict[str, Any] = {}
_JUDGES: Dict[str, Any] = {}


def _make_register(table: Dict[str, Any], kind: str) -> Callable:
    def register(name: str) -> Callable:
        def decorator(obj):
            if name in table and table[name] is not obj:
                raise ValueError(f"Duplicate {kind} name: {name!r}")
            table[name] = obj
            return obj

        return decorator

    return register


register_surface = _make_register(_SURFACES, "surface")
register_adaptor = _make_register(_ADAPTORS, "adaptor")
register_judge = _make_register(_JUDGES, "judge")


def _get(table: Dict[str, Any], name: str, kind: str):
    if name not in table:
        raise KeyError(f"Unknown {kind} {name!r}. Known: {sorted(table)}")
    return table[name]


def get_surface(name: str):
    return _get(_SURFACES, name, "surface")


def get_adaptor(name: str):
    return _get(_ADAPTORS, name, "adaptor")


def get_judge(name: str):
    return _get(_JUDGES, name, "judge")


def surface_names() -> List[str]:
    return sorted(_SURFACES)


def adaptor_names() -> List[str]:
    return sorted(_ADAPTORS)


def judge_names() -> List[str]:
    return sorted(_JUDGES)


def load_all() -> None:
    """Import every module that registers something. Explicit, no discovery."""
    # groupchat and letter are re-export shims now; s7 and s8 register inside
    # generation.register_all() with the other six, so importing them here would be
    # importing a shim for no reason.
    from bench.surfaces import choice, generation  # noqa: F401
    from bench.adaptors import local_hf, opencode  # noqa: F401
    generation.register_all()
