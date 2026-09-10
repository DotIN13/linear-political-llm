"""The legacy ``bench/`` tree was retired, so the parity tests that compare
``bench_v2`` against it cannot run. Skip those tests when ``bench`` is absent
instead of erroring the whole collection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _needs_legacy_bench(path: Path) -> bool:
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(token in src for token in ("from bench.", "from bench import", "import bench\n"))


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("bench") is not None:
        return
    skip = pytest.mark.skip(reason="legacy bench/ tree removed; parity test needs it")
    for item in items:
        if _needs_legacy_bench(Path(str(item.fspath))):
            item.add_marker(skip)
