"""One component per piece of text a scheme sends, resolvable from a task's j2 files.

A scheme's wording is a set of named components -- the share line, the role, the
memory clause, the acknowledgement, the intent sentence, the request. Each has a
default: a ``.j2`` file inside the scheme's own directory here. A task overrides one
by dropping a file of the same name into a directory of the same name:

    tasks/s9_neighborhood/v2/chat/share.j2
    tasks/s9_neighborhood/v2/agentic/role.j2
    tasks/s9_neighborhood/v2/agentic_live/intent.j2

Three sources, in this order:

1. the task's ``<task_dir>/<scheme>/<name>.j2``, if it exists;
2. an explicit ``style`` value, which is how the eleven earlier tasks word their
   prompts and how s1 v2 and s3 v4/v5 stay reproducible;
3. the default ``<here>/<scheme>/<name>.j2``.

The order matters: a task's own file beats a style dict, so a task can migrate one
component at a time without touching its pilot. And a component with no style key --
the two acknowledgements, the memory clauses -- has only the file path, which is why
they could not be overridden at all before.

The join is done here rather than in the files, so no ``.j2`` has to begin with a
leading space to read correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from bench_v2.helpers.prompts import render

HERE = Path(__file__).resolve().parent

# Which style key, if any, supplies each component. The dicts the earlier tasks pass
# are keyed by concept rather than by file; this is the translation.
STYLE_KEY: Dict[tuple, str] = {
    ("chat", "share"): "chat_share",
    ("agentic", "role"): "agentic_system",
    ("agentic_live", "role"): "agentic_live_system",
    ("agentic_live", "intent"): "live_intent",
}

# Every component a scheme has. A name not in here is a typo, and raises.
COMPONENTS: Dict[str, tuple] = {
    "chat": ("share", "memory_clause", "ack", "portrait", "request"),
    "agentic": ("role", "portrait", "memory_clause", "looked", "request"),
    "agentic_live": ("role", "portrait", "memory_clause", "intent", "request"),
}


class Prompts:
    """Resolves a scheme's components, preferring the task's own files."""

    def __init__(self, task_dir: Optional[str | Path] = None,
                 style: Optional[Dict[str, Any]] = None) -> None:
        self.task_dir = Path(task_dir) if task_dir else None
        self.style: Dict[str, Any] = dict(style or {})

    # --- resolution ---------------------------------------------------------
    def get(self, scheme: str, name: str, **context: Any) -> str:
        """The text of one component."""
        if scheme not in COMPONENTS:
            raise ValueError(f"unknown scheme {scheme!r}; expected {sorted(COMPONENTS)}")
        if name not in COMPONENTS[scheme]:
            raise ValueError(
                f"unknown component {name!r} for scheme {scheme!r}; "
                f"expected {COMPONENTS[scheme]}")
        if self.task_dir is not None:
            own = self.task_dir / scheme / f"{name}.j2"
            if own.is_file():
                return render(own, **context)
        key = STYLE_KEY.get((scheme, name))
        if key and self.style.get(key):
            return str(self.style[key])
        return render(HERE / scheme / f"{name}.j2", **context)

    def has_own(self, scheme: str, name: str) -> bool:
        """Whether the task ships this component itself."""
        return (self.task_dir is not None
                and (self.task_dir / scheme / f"{name}.j2").is_file())

    # --- the composed strings ------------------------------------------------
    def share(self, variant: str = "bare") -> str:
        """The line under the photos. The memory clause is appended, not embedded."""
        text = self.get("chat", "share")
        if variant == "memory":
            text += " " + self.get("chat", "memory_clause")
        return text

    def role(self, scheme: str, variant: str = "bare", portrait: Optional[str] = None) -> str:
        """The agent's role text. Portrait sentence, then the memory clause."""
        text = self.get(scheme, "role")
        if portrait:
            text += " " + self.get(scheme, "portrait")
        if variant == "memory":
            text += " " + self.get(scheme, "memory_clause")
        return text

    def request(self, scheme: str, question: str) -> str:
        """The request. The default is the question untouched; a task may frame it.

        Applied in ``build_base`` before the photos/no_photos branch, so the baseline
        is framed too and stays a control for the persona rather than the wording.
        """
        return self.get(scheme, "request", question=question)


def resolve(prompts: Optional["Prompts"] = None, style: Optional[Dict[str, Any]] = None,
            task_dir: Optional[str | Path] = None) -> "Prompts":
    """Whichever resolver the caller already has, or one built from these."""
    return prompts if prompts is not None else Prompts(task_dir=task_dir, style=style)


# --- the default components, as values ----------------------------------------
# Derived from the .j2 files rather than written twice, so the file is the single
# source and these are a convenience for tests and for the compatibility surface.
DEFAULT_CHAT_SHARE = render(HERE / "chat" / "share.j2")
CHAT_ACK = render(HERE / "chat" / "ack.j2")
ME_SENTENCE = render(HERE / "agentic" / "portrait.j2")
MEMORY_CLAUSE_CHAT = render(HERE / "chat" / "memory_clause.j2")
MEMORY_CLAUSE_AGENTIC = render(HERE / "agentic" / "memory_clause.j2")
LOOKED_ACK = render(HERE / "agentic" / "looked.j2")
DEFAULT_AGENTIC_SYSTEM = render(HERE / "agentic" / "role.j2")
DEFAULT_AGENTIC_LIVE_SYSTEM = render(HERE / "agentic_live" / "role.j2")
DEFAULT_LIVE_INTENT = render(HERE / "agentic_live" / "intent.j2")


def style_of(style: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The style dict with a default under every key it knows about."""
    style = dict(style or {})
    for key, default in (("chat_share", DEFAULT_CHAT_SHARE),
                         ("agentic_system", DEFAULT_AGENTIC_SYSTEM),
                         ("agentic_live_system", DEFAULT_AGENTIC_LIVE_SYSTEM),
                         ("live_intent", DEFAULT_LIVE_INTENT)):
        style.setdefault(key, default)
    return style


# The wording the fallback used to be, frozen: s1 v2 and s3 v4/v5 passed no style,
# so these exact strings are in their transcripts. Do not use it for anything new.
LEGACY_NEWS_DIGEST_STYLE: Dict[str, Any] = {
    "agentic_live_system": (
        "You are the user's news digest agent. You have access to the user's "
        "memory directories: /memory/hometown holds photos of where they live, "
        "and /memory/preferences holds photos of things they like. Use the "
        "list_dir and view_image tools to look through them when it helps you "
        "answer the user's request."),
    "live_intent": "I'll look at your memory and recommend news based on your taste.",
}


def defaults() -> Dict[str, Dict[str, str]]:
    """Every default component, for a test that renders the whole set."""
    out: Dict[str, Dict[str, str]] = {}
    for scheme, names in COMPONENTS.items():
        out[scheme] = {}
        for name in names:
            out[scheme][name] = render(HERE / scheme / f"{name}.j2", question="{{question}}")
    return out
