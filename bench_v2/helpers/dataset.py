"""Item loading for pilots.

The sampler itself still lives in ``bench/sample.py``; bench_v2 consumes its
output. This module is the one place a pilot turns an items file into ``Item``
objects, so a missing file degrades to synthetic no-image items in exactly one
way instead of in every pilot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from bench_v2.helpers.prompts import load_jsonl, load_pool
from bench_v2.types import Item, baseline_item


def read_items(path: str | Path, limit: int = 0, split: str = "explore") -> tuple[list[Item], bool]:
    """Items from an items file, or synthetic no-image ones when it is not there.

    Returns ``(items, synthetic)``. ``limit=0`` means every row.
    """
    path = Path(path)
    if not path.exists():
        return [baseline_item() for _ in range(limit or 1)], True
    rows = [row for row in load_jsonl(path) if row.get("split", "explore") == split]
    items = [Item.from_dict(row) for row in rows]
    return (items[:limit] if limit else items), False


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Raw rows, when a pilot needs fields ``Item`` does not carry."""
    path = Path(path)
    return load_jsonl(path) if path.exists() else []


def load_pool_file(path: str | Path, id_field: str = "id",
                   required: Sequence[str] = ()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A ``.jsonl`` pool plus its ``.meta.json`` header, with a light validation."""
    path = Path(path)
    rows, header = load_pool(path)
    seen: set[Any] = set()
    for row in rows:
        for key in required:
            if not row.get(key):
                raise ValueError(f"{path}: {row.get(id_field)!r} is missing {key!r}")
        if row.get(id_field) in seen:
            raise ValueError(f"{path}: duplicate {id_field} {row.get(id_field)!r}")
        seen.add(row.get(id_field))
    return rows, header
