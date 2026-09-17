"""The live agentic arm: the request first, the search in response to it.

The realistic order. The user asks; the agent announces what it is about to do (in
the memory variant only, and before the first tool call); the machine inserts the
search; and the model's generation after the last tool result is the answer, so the
episode reads as one turn the way a modern coding agent runs.

It shares ``agentic``'s wording and its tool chain, and differs only in where the
request sits. That is the contrast, so the two must not drift in anything else.
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
    messages = [
        parts.system_message(p.role("agentic_live", variant, portrait)),
        parts.user_text(question),
    ]
    if variant == "memory":
        messages.append(parts.assistant_text(p.get("agentic_live", "intent")))
    messages += parts.tool_walk(parts.memory_files(n_files, portrait, portrait_name),
                                image_paths, portrait)
    return messages, parts.TOOLS
