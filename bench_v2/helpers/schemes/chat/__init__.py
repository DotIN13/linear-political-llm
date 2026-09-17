"""The chat arm: the persona's photos, an acknowledgement, then the request.

The photos and the request are deliberately in **different** turns. The model
answers a question that follows its own reply, rather than one attached to the
images, so the persona is something it has already responded to by the time it is
asked. That is also what makes the shared prefix -- everything before the final user
turn -- the natural place for ``s_pre``.

Every string comes from a component, so a task changes the wording by dropping a
``.j2`` into its own ``chat/`` directory and changes this shape only by shipping a
``chat.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from bench_v2.helpers import prompt_parts as parts
from bench_v2.helpers.schemes import components


def build(image_paths: Sequence[str], question: str, n_files: int = 3,
          variant: str = "bare", portrait: Optional[str] = None,
          portrait_name: str = parts.ME_FILE, *,
          prompts: Optional[components.Prompts] = None,
          style: Optional[Dict[str, Any]] = None
          ) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    p = components.resolve(prompts, style)
    first = parts.image_turn(image_paths, portrait, p.share(variant),
                               p.get("chat", "portrait") if portrait else "")
    return [
        {"role": "user", "content": first},
        parts.assistant_text(p.get("chat", "ack")),
        parts.user_text(question),
    ], None
