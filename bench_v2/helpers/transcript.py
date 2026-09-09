"""How the persona is delivered: the chat scheme and the agentic scheme.

The whole conversation structure lives in ``templates/chat.j2`` and
``templates/agentic.j2``, one JSON object per line. This module only fills in the
data a template cannot know: the images, the question, the persona clause, and
which files sit in which memory directory. Roles, turn order and wording are the
template's.

The shared prefix -- everything before the final question -- is byte-identical
across all questions within a scheme, which is what makes ``s_pre`` comparable
across questions *by construction*. The parity test builds every scheme x clause
and compares to the old ``bench`` transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench_v2.helpers.prompts import render

_TEMPLATES = Path(__file__).resolve().parent / "templates"
PERSONA_CLAUSES = ("bare", "memory")

# Two directories rather than one flat /memory/user, so the *names* carry the same
# information the chat share line carries: these images are where the person lives
# and what they like. The split is a label we attach, not a property of the photo.
MEMORY_DIRS = ["/memory/hometown", "/memory/preferences"]

# Fixed filenames, not generated, so a transcript is reproducible from the file.
_FILENAME_POOL = ["img_0417.jpg", "img_0903.jpg", "img_3011.jpg",
                  "img_1188.jpg", "img_2274.jpg", "img_0655.jpg",
                  "img_3902.jpg", "img_1461.jpg", "img_2830.jpg",
                  "img_0247.jpg"]
# Two thirds of the photos go in hometown, rounded -- 3 -> 2+1, 10 -> 7+3.
HOMETOWN_SHARE = 2 / 3


def files_by_dir(n_files: int = 3) -> List[Tuple[str, List[str]]]:
    """Which filenames sit in which memory directory, for ``n_files`` photos."""
    if not 1 <= n_files <= len(_FILENAME_POOL):
        raise ValueError(
            f"n_files={n_files} outside 1..{len(_FILENAME_POOL)}; add names to "
            f"_FILENAME_POOL rather than generating them, so transcripts stay "
            f"reproducible from the file"
        )
    n_home = round(n_files * HOMETOWN_SHARE)
    n_home = max(1, min(n_home, n_files - 1)) if n_files > 1 else 1
    names = _FILENAME_POOL[:n_files]
    out = [(MEMORY_DIRS[0], names[:n_home])]
    if n_files > 1:
        out.append((MEMORY_DIRS[1], names[n_home:]))
    return out


FILES_BY_DIR = files_by_dir(3)
FILENAMES = [f for _dir, names in FILES_BY_DIR for f in names]
FILENAMES_LINE = "  ".join(FILENAMES)


TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "List the files in a directory.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Directory to list."}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "view_image", "description": "Open an image file and return its contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Image file to open."}},
                       "required": ["path"]}}},
]


def _messages(path: Path, **context: Any) -> List[Dict[str, Any]]:
    """Render a template to one JSON message per line."""
    rendered = render(path, **context)
    return [json.loads(line) for line in rendered.splitlines() if line.strip()]


def _chat_messages(image_paths: Sequence[str], question: str,
                   clause: str = "bare") -> List[Dict[str, Any]]:
    if clause not in PERSONA_CLAUSES:
        raise ValueError(f"unknown persona clause {clause!r}; expected {PERSONA_CLAUSES}")
    return _messages(_TEMPLATES / "chat.j2", image_paths=list(image_paths),
                     question=question, clause=clause)


def _agentic_messages(image_paths: Sequence[str], question: str,
                      n_files: int = 3, clause: str = "bare") -> List[Dict[str, Any]]:
    if clause not in PERSONA_CLAUSES:
        raise ValueError(f"unknown persona clause {clause!r}; expected {PERSONA_CLAUSES}")
    dirs = files_by_dir(n_files)
    files_flat = [[directory, fname] for directory, names in dirs for fname in names]
    return _messages(_TEMPLATES / "agentic.j2", image_paths=list(image_paths),
                     question=question, clause=clause, dirs=dirs, files_flat=files_flat)


def build_scheme_messages(scheme: str, image_paths: Sequence[str], question: str,
                          n_files: Optional[int] = None, clause: str = "bare"):
    """Build one scheme's message list, or raise on an unknown scheme."""
    n = n_files if n_files is not None else (len(image_paths) or 3)
    if scheme == "chat":
        return _chat_messages(image_paths, question, clause), None
    if scheme == "agentic":
        return _agentic_messages(image_paths, question, n, clause), TOOLS
    raise ValueError(f"unknown scheme {scheme!r}")
