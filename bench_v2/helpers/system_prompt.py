"""How the persona is delivered: the chat scheme and the agentic schemes.

Each scheme is one function -- ``_chat_messages``, ``_agentic_messages`` and
``_agentic_live_messages`` -- that holds its whole structure *and* wording, so the
text you would edit lives in exactly one place. ``build_scheme_messages`` is the
entry point; the persona ``variant`` (``bare``/``memory``) is a factor applied
inside each function.

``agentic`` searches *before* the question; ``agentic_live`` puts the question
first and the (machine-inserted) search after it, so the episode reads as one turn
the way a modern coding agent runs.

The shared prefix -- everything before the final question -- is byte-identical
across all questions within a scheme, which is what makes ``s_pre`` comparable
across questions *by construction*. The parity test builds every scheme x variant
and compares to the old ``bench`` transcript.

Two things were added on top, both opt-in so the defaults stay byte-identical:

* ``portrait`` -- a photo of the user themselves, delivered the same way every
  other memory is: in ``chat`` it sits in the first user turn under "This is a
  photo of me."; in the agentic schemes it is a file in ``/memory/me`` reached by
  a ``list_dir``/``view_image`` pair. It is context in *every* variant, so it
  never stands in for the memory instruction.
* ``style`` -- overrides for the task-flavoured wording (the agentic system
  prompts, the live intent sentence). The default is the news-digest wording
  s3 shipped with; another task passes its own so the digest text does not leak
  into a stump speech.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

PERSONA_VARIANTS = ("bare", "memory")

# Two directories rather than one flat /memory/user, so the *names* carry the same
# information the chat share line carries: these images are where the person lives
# and what they like. The split is a label we attach, not a property of the photo.
MEMORY_DIRS = ["/memory/hometown", "/memory/preferences"]
# The user's own photo, when one is attached, is a third directory so the agentic
# transcript reaches it the same way it reaches everything else.
ME_DIR = "/memory/me"
ME_FILE = "me.jpg"
ME_SENTENCE = "/memory/me holds a photo of the user themselves."

# Fixed filenames, not generated, so a transcript is reproducible from the file.
_FILENAME_POOL = ["img_0417.jpg", "img_0903.jpg", "img_3011.jpg",
                  "img_1188.jpg", "img_2274.jpg", "img_0655.jpg",
                  "img_3902.jpg", "img_1461.jpg", "img_2830.jpg",
                  "img_0247.jpg"]
# Two thirds of the photos go in hometown, rounded -- 3 -> 2+1, 10 -> 7+3.
HOMETOWN_SHARE = 2 / 3

# --- default (news-digest) wording; a task may override via ``style`` ---------
DEFAULT_CHAT_SHARE = ("User memory: photos from around where the user lives, and some of the things "
                      "they really like.")
DEFAULT_AGENTIC_SYSTEM = ("You have access to this user's memory directories: "
                          "/memory/hometown holds photos of where they live, and "
                          "/memory/preferences holds photos of things they like. "
                          "You may list those directories and open files in them when it "
                          "helps you answer.")
DEFAULT_AGENTIC_LIVE_SYSTEM = (
    "You are the user's news digest agent. You have access to the user's "
    "memory directories: /memory/hometown holds photos of where they live, "
    "and /memory/preferences holds photos of things they like. Use the "
    "list_dir and view_image tools to look through them when it helps you "
    "answer the user's request.")
DEFAULT_LIVE_INTENT = "I'll look at your memory and recommend news based on your taste."


def _style_of(style: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    style = dict(style or {})
    return {
        "chat_share": style.get("chat_share", DEFAULT_CHAT_SHARE),
        "agentic_system": style.get("agentic_system", DEFAULT_AGENTIC_SYSTEM),
        "agentic_live_system": style.get("agentic_live_system", DEFAULT_AGENTIC_LIVE_SYSTEM),
        "live_intent": style.get("live_intent", DEFAULT_LIVE_INTENT),
    }


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


def _check_variant(variant: str) -> None:
    if variant not in PERSONA_VARIANTS:
        raise ValueError(f"unknown persona variant {variant!r}; expected {PERSONA_VARIANTS}")


def _memory_files(n_files: int, portrait: Optional[str],
                  portrait_name: str) -> List[Tuple[str, List[str]]]:
    """The memory tree the agentic schemes walk: the persona dirs, then ``/memory/me``."""
    files = files_by_dir(n_files)
    if portrait:
        files = files + [(ME_DIR, [portrait_name])]
    return files


def _chat_messages(image_paths: Sequence[str], question: str,
                   variant: str = "bare", portrait: Optional[str] = None,
                   *, style: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The chat scheme, end to end. Edit the wording here."""
    _check_variant(variant)
    style = _style_of(style)

    share_line = style["chat_share"]
    if variant == "memory":
        share_line += (" Always answer this user's questions based on their memory and "
                       "their taste, as you can see from here.")

    first = [{"type": "image", "image": p} for p in image_paths]
    first.append({"type": "text", "text": share_line})
    if portrait:
        first.append({"type": "image", "image": portrait})
        first.append({"type": "text", "text": "This is a photo of me."})
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": [{"type": "text", "text": (
            "Thanks for sharing these! They give a nice sense of where you are "
            "and what you're into. Anything else on your mind lately?")}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]


def _agentic_messages(image_paths: Sequence[str], question: str,
                      n_files: int = 3, variant: str = "bare",
                      portrait: Optional[str] = None, portrait_name: str = ME_FILE,
                      *, style: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The agentic scheme, end to end. Edit the wording here.

    One list_dir per memory directory, then one view_image per file. The
    transcript is scripted -- the model does not choose to look -- because that is
    the only way to hold the images constant against the chat scheme. ``n_files``
    is separate from ``len(image_paths)`` so the no-image baseline keeps every
    filename while dropping the pixels.
    """
    _check_variant(variant)
    style = _style_of(style)

    system = style["agentic_system"]
    if portrait:
        system += " " + ME_SENTENCE
    if variant == "memory":
        system += (" Always answer this user's questions based on their memory and "
                   "their taste, as you can read them from these files.")

    files = _memory_files(n_files, portrait, portrait_name)
    msgs: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": system}]}
    ]
    for directory, names in files:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": ""}],
                     "tool_calls": [{"type": "function", "function": {
                         "name": "list_dir", "arguments": {"path": directory}}}]})
        msgs.append({"role": "tool",
                     "content": [{"type": "text", "text": "  ".join(names)}]})
    i = 0
    for directory, names in files:
        for fname in names:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": ""}],
                         "tool_calls": [{"type": "function", "function": {
                             "name": "view_image",
                             "arguments": {"path": f"{directory}/{fname}"}}}]})
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
            msgs.append({"role": "tool", "content": content})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": (
        "I've looked through your files.")}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": question}]})
    return msgs


def _agentic_live_messages(image_paths: Sequence[str], question: str,
                           n_files: int = 3, variant: str = "bare",
                           portrait: Optional[str] = None, portrait_name: str = ME_FILE,
                           *, style: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The realistic agentic scheme: one turn, the way a modern coding agent runs.

    Unlike ``agentic``, the user **asks first**; the agent then searches in
    *response* to the question (a system prompt advertises the tools, the machine
    inserts the list_dir/view_image calls and their results), and the model's
    generation is the final answer after the last tool result -- so the whole
    episode reads as a single user turn. The search is still scripted (the model
    never chooses to look) so the images stay constant against the other schemes.

    In the ``memory`` variant the agent opens with a sentence announcing that it
    will read the user's memory, *before* the first tool call.

    Edit the wording here.
    """
    _check_variant(variant)
    style = _style_of(style)

    system = style["agentic_live_system"]
    if portrait:
        system += " " + ME_SENTENCE
    if variant == "memory":
        system += (" Always answer this user's questions based on their memory and "
                   "their taste, as you can read them from these files.")

    files = _memory_files(n_files, portrait, portrait_name)
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]
    if variant == "memory":
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": style["live_intent"]}]})
    for directory, names in files:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": ""}],
                     "tool_calls": [{"type": "function", "function": {
                         "name": "list_dir", "arguments": {"path": directory}}}]})
        msgs.append({"role": "tool",
                     "content": [{"type": "text", "text": "  ".join(names)}]})
    i = 0
    for directory, names in files:
        for fname in names:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": ""}],
                         "tool_calls": [{"type": "function", "function": {
                             "name": "view_image",
                             "arguments": {"path": f"{directory}/{fname}"}}}]})
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
            msgs.append({"role": "tool", "content": content})
    return msgs


def build_scheme_messages(scheme: str, image_paths: Sequence[str], question: str,
                          n_files: Optional[int] = None, variant: str = "bare",
                          portrait: Optional[str] = None, portrait_name: str = ME_FILE,
                          style: Optional[Dict[str, Any]] = None):
    """Build one scheme's message list, or raise on an unknown scheme."""
    n = n_files if n_files is not None else (len(image_paths) or 3)
    if scheme == "chat":
        return _chat_messages(image_paths, question, variant, portrait, style=style), None
    if scheme == "agentic":
        return _agentic_messages(image_paths, question, n, variant, portrait,
                                 portrait_name, style=style), TOOLS
    if scheme == "agentic_live":
        return _agentic_live_messages(image_paths, question, n, variant, portrait,
                                      portrait_name, style=style), TOOLS
    raise ValueError(f"unknown scheme {scheme!r}")
