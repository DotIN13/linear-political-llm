"""Helpers for assembling a prompt: render a task's ``.j2`` file, and put images and
text into one user turn. The wording lives in the task's files, not here."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from jinja2 import StrictUndefined, Template

from bench_llm.types import Conversation

_CACHE: Dict[Path, Template] = {}


def render(path: str | Path, **context: Any) -> str:
    """The template at ``path``; a variable the template names but is not given raises."""
    path = Path(path).resolve()
    if path not in _CACHE:
        _CACHE[path] = Template(path.read_text(encoding="utf-8"), keep_trailing_newline=False,
                                undefined=StrictUndefined)
    return _CACHE[path].render(**context)


def user_turn(text: str, images: Sequence[str] = (), images_first: bool = True) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [{"type": "image", "image": p} for p in images]
    t = {"type": "text", "text": text}
    return {"role": "user", "content": parts + [t] if images_first else [t] + parts}


def conversation(text: str, images: Sequence[str] = (), system: Optional[str] = None,
                 images_first: bool = True) -> Conversation:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append(user_turn(text, images, images_first))
    return Conversation(messages=messages)
