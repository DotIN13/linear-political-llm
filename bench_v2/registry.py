"""Adaptor registry. Explicit imports, no discovery.

The old bench had one registry for surfaces, adaptors and judges. bench_v2 has
no surfaces to register -- a pilot is the entry point -- so this is only the
decorator the adaptor modules use to name themselves.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

_ADAPTORS: Dict[str, Any] = {}


def register_adaptor(name: str) -> Callable:
    def decorator(obj):
        if name in _ADAPTORS and _ADAPTORS[name] is not obj:
            raise ValueError(f"Duplicate adaptor name: {name!r}")
        _ADAPTORS[name] = obj
        return obj

    return decorator


def get_adaptor(name: str):
    if name not in _ADAPTORS:
        raise KeyError(f"Unknown adaptor {name!r}. Known: {sorted(_ADAPTORS)}")
    return _ADAPTORS[name]


def adaptor_names() -> List[str]:
    return sorted(_ADAPTORS)


def load_adaptors() -> None:
    """Import every adaptor module so its decorator runs."""
    from bench_v2.adaptors import base, local_hf, opencode, vllm_server  # noqa: F401
