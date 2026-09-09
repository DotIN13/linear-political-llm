"""Prompt files, rendered as Jinja2.

Every task keeps its wording in files beside its pilots rather than in Python.
A constant ask is still a template -- the file *is* the prompt -- so there is one
path for all of them and no "is this one constant?" branch.

``keep_trailing_newline=False`` drops the file's own final newline, matching the
old ``bench.surfaces.shared.prompts.text`` loader exactly, so a prompt moved from
``bench`` to ``bench_v2`` renders byte-identically. ``StrictUndefined`` makes a
mistyped variable raise instead of rendering empty -- the difference between a
red test and a prompt with a hole in it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template

_TEMPLATES: dict[Path, Template] = {}


def render(path: str | Path, **context: Any) -> str:
    """Render the template at ``path`` with ``context``."""
    path = Path(path).resolve()
    if path not in _TEMPLATES:
        _TEMPLATES[path] = Template(
            path.read_text(encoding="utf-8"), keep_trailing_newline=False,
            trim_blocks=False, lstrip_blocks=False, undefined=StrictUndefined,
        )
    return _TEMPLATES[path].render(**context)


def load_text(path: str | Path) -> str:
    """A non-template file verbatim, with exactly one trailing newline removed."""
    raw = Path(path).read_text(encoding="utf-8")
    return raw[:-1] if raw.endswith("\n") else raw


def load_json(path: str | Path) -> dict[str, Any]:
    """A small json file, for text whose whitespace is significant (a prefill)."""
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pool(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A ``.jsonl`` pool plus its sibling ``.meta.json`` header (``{}`` if absent)."""
    path = Path(path)
    rows = load_jsonl(path)
    header = path.with_suffix(".meta.json") if path.suffix == ".jsonl" else path
    meta = load_json(header) if header.exists() else {}
    return rows, meta
