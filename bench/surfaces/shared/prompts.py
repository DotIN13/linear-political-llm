"""Loading a task's prompt files.

Every task keeps its wording in ``tasks/<task>/prompts/`` rather than in Python, so
that changing what the model is asked does not mean editing code. This module is the
only thing that reads those files, and it is deliberately small: four loaders and one
rule each.

**The format is chosen per task, not once for all of them.** A constant ask is a
``.txt`` because that is the honest representation of a constant string; the two tasks
that build a prompt out of parts get a ``.j2``; the three that carry an item pool get a
``.jsonl`` and a ``.meta.json``. Rendering a variable-free string through a template
engine would buy nothing and cost a whitespace hazard.

**Byte-identity is the whole point.** ``bench/tests/test_prompts_golden.py`` compares
every rendered prompt against a snapshot taken before any of this moved, so a stray
trailing newline in a template is a red test rather than a silent change to the
instrument.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

# Imported at module scope on purpose. Every task imports this module, so a missing
# jinja2 fails at import time with an obvious message, rather than half way through a
# run when s3 or s7 first renders. It is in requirements.txt; it has to be installed
# on midway3 and DeltaAI before the next run.
from jinja2 import StrictUndefined, Template

_TEMPLATES: Dict[str, Template] = {}


def _dir(entry: str) -> str:
    """The ``prompts/`` directory beside a task's entry file. Pass ``__file__``."""
    return os.path.join(os.path.dirname(os.path.abspath(entry)), "prompts")


def text(entry: str, name: str = "ask.txt") -> str:
    """A constant prompt, verbatim. **The file is the prompt.**

    Exactly one trailing newline is removed -- every editor writes one and no ask ends
    in a line break -- and nothing else is touched: no dedent, no unwrapping, no
    stripping. So an ask that is one long line is stored as one long line, however
    awkward that is to look at, because anything else would mean the file and the
    prompt are not the same string.

    Text whose *trailing* whitespace carries meaning must not come through here -- a
    prefill ends in a blank line, and two invisible bytes at the end of a file is a bug
    waiting to happen. Use ``strings`` for those, where the escapes are visible.
    """
    with open(os.path.join(_dir(entry), name), encoding="utf-8") as handle:
        raw = handle.read()
    return raw[:-1] if raw.endswith("\n") else raw


def strings(entry: str, name: str) -> Dict[str, str]:
    """Prompt strings from a small json file, for text whose whitespace is significant.

    ``"Here's an outline for your stump speech:\\n\\n"`` is a prefill, and that trailing
    blank line is load-bearing -- it is what makes the model continue an outline rather
    than start a turn. In json you can see it; in a ``.txt`` you cannot.
    """
    with open(os.path.join(_dir(entry), name), encoding="utf-8") as handle:
        return dict(json.load(handle))


def template(entry: str, name: str) -> Template:
    """A jinja template, for the two tasks that assemble a prompt out of parts.

    ``keep_trailing_newline=False`` drops the file's own final newline, and
    ``trim_blocks``/``lstrip_blocks`` are both off, so the template text is emitted
    exactly as written and a newline inside an ``{% if %}`` is a newline in the output.
    ``StrictUndefined`` makes a mistyped variable raise instead of rendering empty --
    which is the difference between a red test and a prompt with a hole in it.
    """
    path = os.path.join(_dir(entry), name)
    if path not in _TEMPLATES:
        with open(path, encoding="utf-8") as handle:
            _TEMPLATES[path] = Template(
                handle.read(), keep_trailing_newline=False,
                trim_blocks=False, lstrip_blocks=False, undefined=StrictUndefined,
            )
    return _TEMPLATES[path]


def pool(entry: str, name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """A task's item pool: ``<name>.jsonl`` rows, plus its ``<name>.meta.json`` header.

    Rows come back **in file order**, which is load-bearing: the surfaces fingerprint
    their pool by hashing row order, and that fingerprint is recorded in every trial's
    ``meta["dataset"]``.

    The header is a separate file because jsonl has no room for one, and because it is
    not prompt material: it is the design record (``version``, ``design``,
    ``framing_rule``) and, for s3, the analysis constants the DV is read against
    (``set_mean_slant``, ``null_centered_pick5`` and the rest). ``version`` in
    particular is recorded in every trial, so dropping it would not be a tidy-up.
    """
    base = _dir(entry)
    rows: List[Dict[str, Any]] = []
    with open(os.path.join(base, f"{name}.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    with open(os.path.join(base, f"{name}.meta.json"), encoding="utf-8") as handle:
        meta = dict(json.load(handle))
    return rows, meta
