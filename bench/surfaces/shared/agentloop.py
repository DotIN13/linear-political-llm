"""Drive a surface's tools until it answers, or until it has had enough turns.

Every other surface in this project is one request and one reply. s15_shopping is
an environment: the model searches, sees results, searches again, and only then
recommends. That needs a loop, and the loop needs to be somewhere both the pilot
and any later runner can use.

**It is deliberately small and deliberately strict.** It executes only tools the
surface offers, it stops the moment the surface says it has an answer, and it caps
the turns. A loop that runs away is a loop that spends a GPU allocation on one
trial.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

MAX_TURNS = 8


def _arguments(call: Dict[str, Any]) -> Dict[str, Any]:
    """Tool arguments come back as a JSON *string*. A malformed one is empty, not a
    crash -- the model failing to format arguments is data, not an exception."""
    raw = ((call.get("function") or {}).get("arguments")) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:                                       # noqa: BLE001
        return {}


def run_agent(adaptor, trial, surface, *, terminal: str,
              max_turns: int = MAX_TURNS) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Run the tool loop.

    Returns ``(calls, transcript, error)``:

    * ``calls`` -- every tool call in order, as ``{"name", "arguments"}``. This is
      what the surface reads its DV from.
    * ``transcript`` -- the messages as they grew, so a record can be re-read later
      without re-running anything.
    * ``error`` -- the first adaptor error, or None.

    Stops on ``terminal`` (the tool that means "answered"), on a reply with no tool
    calls, or at ``max_turns``. **A run that hits the cap is not an answer** and the
    surface will see no terminal call, which is what it should see.
    """
    messages = [dict(m) for m in trial.conversation.messages]
    calls: List[Dict[str, Any]] = []

    for _ in range(max_turns):
        # Trial and Conversation are frozen dataclasses, so each turn is a new
        # object rather than a mutated one. That is a feature here: the record can
        # keep the transcript it actually sent.
        step = replace(trial, conversation=replace(trial.conversation, messages=list(messages)))
        resp = adaptor.run(step)
        if resp.error:
            return calls, messages, resp.error

        raw_calls = (resp.usage or {}).get("raw_tool_calls") or []
        if not raw_calls:
            messages.append({"role": "assistant",
                             "content": [{"type": "text", "text": resp.text or ""}]})
            return calls, messages, None

        messages.append({"role": "assistant", "content": [{"type": "text", "text": resp.text or ""}],
                         "tool_calls": raw_calls})
        hit_terminal = False
        for rc in raw_calls:
            name = (rc.get("function") or {}).get("name")
            args = _arguments(rc)
            calls.append({"name": name, "arguments": args})
            if name == terminal:
                hit_terminal = True
                continue
            result = surface.call(name, args, int(trial.meta.get("price_rotation", 0)))
            messages.append({"role": "tool", "tool_call_id": rc.get("id") or name,
                             "name": name, "content": result})
        if hit_terminal:
            return calls, messages, None

    return calls, messages, None
