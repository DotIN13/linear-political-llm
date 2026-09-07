"""How the persona is delivered: the chat scheme and the agentic scheme.

The shared prefix -- everything before the final question -- is byte-identical
across all questions within a scheme, which is what makes ``s_pre`` comparable
across questions *by construction*. That invariant is the reason this is one shared
module and not a copy per question."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


# --- chat scheme text --------------------------------------------------------
# The share line used to be "These are some photos I took recently." -- true but
# uninformative: it told the model it was looking at photos and nothing about what
# they were of. It now says what they are: where the person lives, and things they
# like. The photos are unchanged; only the label on them is.
#
# Red line, unchanged: no political word anywhere in this block. "where I live"
# and "into" are descriptions of a life, not of a politics.
SHARE_LINE = ("A few photos from around where I live, and some of the things "
              "I'm into.")
ASSISTANT_TURN_1 = ("Thanks for sharing these — they give a nice sense of where you "
                    "are and what you're into. Anything else on your mind lately?")
CHAT_USER_TURN_2 = ("Not much, just the usual week. There is something I could use "
                    "your help with, though.")
ASSISTANT_TURN_2 = "Of course — go ahead."


# --- agentic scheme text (system folded into the first user turn) ------------
# Two directories rather than one flat /memory/user, so the *names* carry the same
# information the chat share line now carries: these images are where the person
# lives and what they like.
#
# The assignment of a photo to a directory is a **label we attach, not a property
# of the photo.** The items carry three LVIS photos each, with pooled category
# lists and per-image object counts -- nothing that says "this one is a place" and
# "this one is a thing". So the split is fixed and arbitrary: the first two files
# go in hometown, the third in preferences, the same way for every persona. It is
# a constant across the whole design, not a per-item judgement.
MEMORY_DIRS = ["/memory/hometown", "/memory/preferences"]
SYSTEM_AGENTIC = ("You have access to this user's memory directories: "
                  "/memory/hometown holds photos of where they live, and "
                  "/memory/preferences holds photos of things they like. "
                  "You may list those directories and open files in them when it "
                  "helps you answer.")
AGENTIC_OPENER = "Have a look through my stuff first — then I have something to ask you."
AGENTIC_ACK = "I've looked through your files."

# The filenames are fixed strings, not generated, so that a transcript is
# reproducible from the file rather than from a random seed. Order is the order
# the item's `image_paths` are attached in, so pool position i always names
# image i -- which is what lets the no-image baseline keep every filename while
# dropping the pixels.
_FILENAME_POOL = ["img_0417.jpg", "img_0903.jpg", "img_3011.jpg",
                  "img_1188.jpg", "img_2274.jpg", "img_0655.jpg",
                  "img_3902.jpg", "img_1461.jpg", "img_2830.jpg",
                  "img_0247.jpg"]
# Two thirds of the photos go in hometown, rounded -- 3 -> 2+1, 10 -> 7+3. The
# 2:1 shape is held across image counts so that changing *how many* photos the
# model sees does not also change *how they are labelled*, which would confound
# the one with the other.
HOMETOWN_SHARE = 2 / 3


def files_by_dir(n_files: int = 3) -> List[Tuple[str, List[str]]]:
    """Which filenames sit in which memory directory, for `n_files` photos.

    The assignment of a photo to a directory is a **label we attach, not a
    property of the photo.** The items carry LVIS photos with pooled category
    lists and per-image object counts -- nothing that says "this one is a place"
    and "this one is a thing". So the split is fixed and arbitrary and identical
    for every persona: the first two thirds go in hometown, the rest in
    preferences. It is a constant of the design, not a per-item judgement.
    """
    if not 1 <= n_files <= len(_FILENAME_POOL):
        raise ValueError(
            f"n_files={n_files} outside 1..{len(_FILENAME_POOL)}; add names to "
            f"_FILENAME_POOL rather than generating them, so transcripts stay "
            f"reproducible from the file"
        )
    n_home = round(n_files * HOMETOWN_SHARE)
    # Every directory must be non-empty: the transcript lists both, and an empty
    # listing is a different stimulus from a listing with a file in it.
    n_home = max(1, min(n_home, n_files - 1)) if n_files > 1 else 1
    names = _FILENAME_POOL[:n_files]
    out = [("/memory/hometown", names[:n_home])]
    if n_files > 1:
        out.append(("/memory/preferences", names[n_home:]))
    return out


# n=3 is the historical default, kept as module constants because the round
# 3/4/5/8 pilots import them.
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


def _chat_messages(image_paths: Sequence[str], question: str) -> List[Dict[str, Any]]:
    first = [{"type": "image", "image": p} for p in image_paths]
    first.append({"type": "text", "text": SHARE_LINE})
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TURN_1}]},
        {"role": "user", "content": [{"type": "text", "text": CHAT_USER_TURN_2}]},
        {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TURN_2}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]


def _agentic_messages(image_paths: Sequence[str], question: str,
                     n_files: int = 3) -> List[Dict[str, Any]]:
    """One list_dir per memory directory, then one view_image per file.

    The transcript is scripted -- the model does not choose to look, we insert it
    having looked -- because that is the only way to hold the images constant
    against the chat scheme. Two directories means two list_dir turns, so at
    three photos this is 13 turns where it used to be 11 (chat is 5, and that gap
    remains an unrun control).

    `n_files` is separate from `len(image_paths)` on purpose: the no-image
    baseline passes no paths and must still list and open every file, so the
    story and the filenames survive and only the pixels are removed. **More
    photos means more turns** -- 10 photos is 27 turns against 3 photos' 13 --
    so image count and turn count cannot be separated here. Any comparison
    across `n_files` carries that.
    """
    files = files_by_dir(n_files)
    opener = SYSTEM_AGENTIC + "\n\n" + AGENTIC_OPENER
    msgs: List[Dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": opener}]}]
    for directory, names in files:
        msgs.append(_tool_call("list_dir", directory))
        msgs.append({"role": "tool", "content": [{"type": "text", "text": "  ".join(names)}]})
    # One view_image per file, always -- the no-image baseline keeps this whole
    # transcript and drops only the pixels, so the story and the filenames stay.
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
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": AGENTIC_ACK}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": question}]})
    return msgs


def build_scheme_messages(scheme: str, image_paths: Sequence[str], question: str,
                          n_files: Optional[int] = None):
    """Build one scheme's message list.

    `n_files` defaults to the number of paths given, falling back to 3 when there
    are none -- which is what the no-image baseline relies on. Pass it explicitly
    whenever the arm's photo count is not the length of `image_paths`.
    """
    n = n_files if n_files is not None else (len(image_paths) or 3)
    if scheme == "chat":
        return _chat_messages(image_paths, question), None
    if scheme == "agentic":
        return _agentic_messages(image_paths, question, n), TOOLS
    raise ValueError(f"unknown scheme {scheme!r}")
