"""Helpers for building an item pool. Which files, which fields and which labels
belong to a benchmark is the pilot's business; these only read, join and sample.

    rows = read_csv(path, numeric=("image_mean",))
    rows = dedupe(rows, "record_id")
    stratify(rows, "image_mean", strata=10)
    rows = sample_per_stratum(rows, per_stratum=50, seed=42)
    labels = read_table(labels_csv, key="bioguide")
    items = to_items(rows, id_field="record_id", path_field="image_path", root=repo)
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from bench_llm.types import Item

csv.field_size_limit(10_000_000)


def repo_root() -> Path:
    """``LPL_REPO_ROOT``, or the directory that holds this package."""
    env = os.environ.get("LPL_REPO_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[1]


def resolve(path: str, root: Optional[str | Path] = None) -> str:
    return path if os.path.isabs(path) else str(Path(root or repo_root()) / path)


def file_sha16(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def read_csv(path: str | Path, numeric: Sequence[str] = (),
             keep: Optional[Callable[[Dict[str, str]], bool]] = None) -> List[Dict[str, Any]]:
    """Rows as dicts; ``numeric`` fields cast to float; ``keep`` filters raw rows."""
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if keep and not keep(raw):
                continue
            row: Dict[str, Any] = dict(raw)
            for f in numeric:
                row[f] = float(raw[f]) if raw.get(f) not in (None, "") else None
            rows.append(row)
    return rows


def dedupe(rows: Iterable[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """First row per key, in order."""
    seen, out = set(), []
    for r in rows:
        if r[key] not in seen:
            seen.add(r[key])
            out.append(r)
    return out


def stratify(rows: List[Dict[str, Any]], field: str, strata: int, name: str = "stratum") -> None:
    """Equal-count bins of ``field``, 0 = lowest, written into each row as ``name``."""
    order = sorted(range(len(rows)), key=lambda i: rows[i][field])
    for rank, i in enumerate(order):
        rows[i][name] = min(strata - 1, rank * strata // max(1, len(rows)))


def sample_per_stratum(rows: Sequence[Dict[str, Any]], per_stratum: int, seed: int,
                       name: str = "stratum", order_by: Optional[str] = None) -> List[Dict[str, Any]]:
    """``per_stratum`` rows from each stratum with a fixed seed; 0 keeps every row."""
    if per_stratum <= 0:
        return list(rows)
    rng = random.Random(seed)
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r[name], []).append(r)
    out = []
    for s in sorted(groups):
        pool = list(groups[s])
        rng.shuffle(pool)
        pick = pool[:per_stratum]
        out.extend(sorted(pick, key=lambda r: r[order_by]) if order_by else pick)
    return out


def read_table(path: str | Path, key: str, numeric: Sequence[str] = (),
               latest_by: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """``key`` -> row. With ``latest_by``, the row with the largest value of it wins."""
    table: Dict[str, Dict[str, Any]] = {}
    for row in read_csv(path, numeric=numeric):
        k = row.get(key)
        if not k:
            continue
        if latest_by is None or k not in table or str(row[latest_by]) >= str(table[k][latest_by]):
            table[k] = row
    return table


def to_items(rows: Iterable[Dict[str, Any]], *, id_field: str, path_field: str,
             data_fields: Optional[Sequence[str]] = None,
             root: Optional[str | Path] = None) -> List[Item]:
    """One single-image Item per row; ``data_fields`` (default: all) travel in ``item.data``."""
    items = []
    for r in rows:
        data = {k: r[k] for k in (data_fields or r.keys()) if k in r}
        items.append(Item(item_id=str(r[id_field]), image_paths=[resolve(r[path_field], root)], data=data))
    return items
