"""The structural pieces a scheme is assembled from. No wording, no ordering.

Every string a scheme sends comes from a component (``helpers/schemes/components.py``),
which a task can override with a ``.j2`` file. This module holds only the shapes data
takes: which filenames sit in which memory directory, the message constructors, the
scripted tool chain, and the tool definitions.

A scheme's *shape* -- how many turns, which role carries the request, where the
scripted search sits -- is one small file in ``helpers/schemes/``. What those files
compose is here: the photo turn, the share line, the role text, the
``list_dir``/``view_image`` chain, and the two canned acknowledgements.

The split is worth having because the two halves change at different rates. Wording
changes often and belongs to a task (``style``). The tool chain barely changes at
all and must not: every scheme's ``s_pre`` comparability, and the claim that the
agentic arms differ only in where the request sits, rest on the chain being
identical in construction. So the chain is generated in **one** function and a
scheme file calls it rather than writing it out. Before this split the chain
existed once inside a three-scheme module and a task had no way to reuse it without
copying it.

Every function here is pure: it takes what it needs and returns a value. None of
them decide what order the pieces go in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

PERSONA_VARIANTS = ("bare", "memory")

# Two directories rather than one flat /memory/user, so the *names* carry the same
# information the chat share line carries: these images are where the person lives
# and what they like. The split is a label we attach, not a property of the photo.
MEMORY_DIRS = ["/memory/hometown", "/memory/preferences"]
# The user's own photo, when one is attached, is a third directory so the agentic
# transcript reaches it the way it reaches everything else. The sentence under it is
# a component, not a constant here.
ME_DIR = "/memory/me"
ME_FILE = "me.jpg"

# Fixed filenames, not generated, so a transcript is reproducible from the file.
_FILENAME_POOL = ["img_0417.jpg", "img_0903.jpg", "img_3011.jpg",
                  "img_1188.jpg", "img_2274.jpg", "img_0655.jpg",
                  "img_3902.jpg", "img_1461.jpg", "img_2830.jpg",
                  "img_0247.jpg"]
# Two thirds of the photos go in hometown, rounded -- 3 -> 2+1, 10 -> 7+3.
HOMETOWN_SHARE = 2 / 3

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

# --- turns --------------------------------------------------------------------
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


def memory_files(n_files: int, portrait: Optional[str],
                 portrait_name: str = ME_FILE) -> List[Tuple[str, List[str]]]:
    """The memory tree the agentic schemes walk: the persona dirs, then ``/memory/me``."""
    files = files_by_dir(n_files)
    if portrait:
        files = files + [(ME_DIR, [portrait_name])]
    return files


def user_text(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant_text(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def system_message(text: str) -> Dict[str, Any]:
    return {"role": "system", "content": [{"type": "text", "text": text}]}


def tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": ""}],
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def tool_result(content: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"role": "tool", "content": content}


def image_turn(image_paths: Sequence[str], portrait: Optional[str] = None,
               text: str = "", portrait_text: str = "") -> List[Dict[str, Any]]:
    """The first user turn: the photos, the user's own photo, then the line.

    Both strings are passed in: the share line and the sentence under the user's own
    photo are components, so they belong to the task, not here.
    """
    content: List[Dict[str, Any]] = [{"type": "image", "image": p} for p in image_paths]
    content.append({"type": "text", "text": text})
    if portrait:
        content.append({"type": "image", "image": portrait})
        content.append({"type": "text", "text": portrait_text})
    return content


def tool_walk(files: Sequence[Tuple[str, List[str]]], image_paths: Sequence[str],
              portrait: Optional[str] = None) -> List[Dict[str, Any]]:
    """The scripted ``list_dir`` per directory, then ``view_image`` per file.

    The model never chooses to look: the calls and their results are inserted by
    the machine, which is the only way to hold the pixels constant against the chat
    scheme. ``image_paths`` may be shorter than the file list -- that is the
    no-image baseline, which keeps every filename and drops the pixels.
    """
    msgs: List[Dict[str, Any]] = []
    for directory, names in files:
        msgs.append(tool_call("list_dir", {"path": directory}))
        msgs.append(tool_result([{"type": "text", "text": "  ".join(names)}]))
    i = 0
    for directory, names in files:
        for fname in names:
            msgs.append(tool_call("view_image", {"path": f"{directory}/{fname}"}))
            content: List[Dict[str, Any]] = []
            if directory == ME_DIR:
                if portrait:
                    content.append({"type": "image", "image": portrait})
            elif i < len(image_paths):
                content.append({"type": "image", "image": image_paths[i]})
                i += 1
            else:
                i += 1
            content.append({"type": "text", "text": fname})
            msgs.append(tool_result(content))
    return msgs
