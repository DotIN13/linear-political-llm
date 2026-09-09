"""How the persona is delivered: the chat scheme and the agentic scheme.

The shared prefix -- everything before the final question -- is byte-identical
across all questions within a scheme, which is what makes ``s_pre`` comparable
across questions *by construction*. That invariant is the reason this is one shared
module and not a copy per question.

The wording lives in ``templates/transcript.j2``, rendered with the persona
``clause`` (``bare``/``memory``). The message *structure* -- which turns, in which
order, with which images and tool calls -- stays here, because it is code, not
prose. A wording edit is a template edit; the parity test builds every scheme x
clause and compares to the old ``bench`` transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench_v2.helpers.prompts import render

_TEMPLATE = Path(__file__).resolve().parent / "templates" / "transcript.j2"
PERSONA_CLAUSES = ("bare", "memory")


def _strings(clause: str = "bare") -> Dict[str, str]:
    """The transcript wording for one persona clause, from the template."""
    if clause not in PERSONA_CLAUSES:
        raise ValueError(f"unknown persona clause {clause!r}; expected {PERSONA_CLAUSES}")
    return json.loads(render(_TEMPLATE, clause=clause))


def share_line(clause: str = "bare") -> str:
    return _strings(clause)["share_line"]


def system_agentic(clause: str = "bare") -> str:
    return _strings(clause)["system_agentic"]


# Back-compat module constants (clause="bare"); the old round pilots import these.
SHARE_LINE = _strings("bare")["share_line"]
SYSTEM_AGENTIC = _strings("bare")["system_agentic"]
ASSISTANT_TURN_1 = _strings("bare")["assistant_turn_1"]
CHAT_USER_TURN_2 = _strings("bare")["chat_user_turn_2"]
ASSISTANT_TURN_2 = _strings("bare")["assistant_turn_2"]
AGENTIC_OPENER = _strings("bare")["agentic_opener"]
AGENTIC_ACK = _strings("bare")["agentic_ack"]


# --- agentic scheme layout ---------------------------------------------------
# Two directories rather than one flat /memory/user, so the *names* carry the same
# information the chat share line carries: these images are where the person lives
# and what they like. The split is a label we attach, not a property of the photo:
# the first two files go in hometown, the third in preferences, the same way for
# every persona.
MEMORY_DIRS = ["/memory/hometown", "/memory/preferences"]

# The filenames are fixed strings, not generated, so that a transcript is
# reproducible from the file rather than from a random seed. Order is the order
# the item's `image_paths` are attached in, so pool position i always names image i.
_FILENAME_POOL = ["img_0417.jpg", "img_0903.jpg", "img_3011.jpg",
                  "img_1188.jpg", "img_2274.jpg", "img_0655.jpg",
                  "img_3902.jpg", "img_1461.jpg", "img_2830.jpg",
                  "img_0247.jpg"]
# Two thirds of the photos go in hometown, rounded -- 3 -> 2+1, 10 -> 7+3. The
# 2:1 shape is held across image counts so that changing *how many* photos the
# model sees does not also change *how they are labelled*.
HOMETOWN_SHARE = 2 / 3


def files_by_dir(n_files: int = 3) -> List[Tuple[str, List[str]]]:
    """Which filenames sit in which memory directory, for `n_files` photos."""
    if not 1 <= n_files <= len(_FILENAME_POOL):
        raise ValueError(
            f"n_files={n_files} outside 1..{len(_FILENAME_POOL)}; add names to "
            f"_FILENAME_POOL rather than generating them, so transcripts stay "
            f"reproducible from the file"
        )
    n_home = round(n_files * HOMETOWN_SHARE)
    n_home = max(1, min(n_home, n_files - 1)) if n_files > 1 else 1
    names = _FILENAME_POOL[:n_files]
    out = [("/memory/hometown", names[:n_home])]
    if n_files > 1:
        out.append(("/memory/preferences", names[n_home:]))
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


def _tool_call(name: str, path: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [{"type": "function",
                        "function": {"name": name, "arguments": {"path": path}}}],
    }


def _chat_messages(image_paths: Sequence[str], question: str,
                   clause: str = "bare") -> List[Dict[str, Any]]:
    s = _strings(clause)
    first = [{"type": "image", "image": p} for p in image_paths]
    first.append({"type": "text", "text": s["share_line"]})
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": [{"type": "text", "text": s["assistant_turn_1"]}]},
        {"role": "user", "content": [{"type": "text", "text": s["chat_user_turn_2"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": s["assistant_turn_2"]}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]


def _agentic_messages(image_paths: Sequence[str], question: str,
                     n_files: int = 3, clause: str = "bare") -> List[Dict[str, Any]]:
    """One list_dir per memory directory, then one view_image per file.

    The transcript is scripted -- the model does not choose to look -- because
    that is the only way to hold the images constant against the chat scheme.
    `n_files` is separate from `len(image_paths)` so the no-image baseline keeps
    every filename while dropping the pixels.
    """
    s = _strings(clause)
    files = files_by_dir(n_files)
    opener = s["system_agentic"] + "\n\n" + s["agentic_opener"]
    msgs: List[Dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": opener}]}]
    for directory, names in files:
        msgs.append(_tool_call("list_dir", directory))
        msgs.append({"role": "tool", "content": [{"type": "text", "text": "  ".join(names)}]})
    i = 0
    for directory, names in files:
        for fname in names:
            msgs.append(_tool_call("view_image", f"{directory}/{fname}"))
            content: List[Dict[str, Any]] = []
            if i < len(image_paths):
                content.append({"type": "image", "image": image_paths[i]})
            content.append({"type": "text", "text": fname})
            msgs.append({"role": "tool", "content": content})
            i += 1
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": s["agentic_ack"]}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": question}]})
    return msgs


def build_scheme_messages(scheme: str, image_paths: Sequence[str], question: str,
                          n_files: Optional[int] = None, clause: str = "bare"):
    """Build one scheme's message list."""
    n = n_files if n_files is not None else (len(image_paths) or 3)
    if scheme == "chat":
        return _chat_messages(image_paths, question, clause), None
    if scheme == "agentic":
        return _agentic_messages(image_paths, question, n, clause), TOOLS
    raise ValueError(f"unknown scheme {scheme!r}")
