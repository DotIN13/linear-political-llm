"""The agentic arm: the role, the scripted search, then the request.

The request is the **last** message, after the search, and that is this arm's whole
manipulation: the agent has already looked at the user's memory before it is told
which item it is working on. So the role cannot travel with the request -- it would
arrive after the work it was meant to frame -- and here it is the first turn.
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
    messages = [parts.user_text(p.role("agentic", variant, portrait))]
    messages += parts.tool_walk(parts.memory_files(n_files, portrait, portrait_name),
                                image_paths, portrait)
    messages.append(parts.assistant_text(p.get("agentic", "looked")))
    messages.append(parts.user_text(question))
    return messages, parts.TOOLS
