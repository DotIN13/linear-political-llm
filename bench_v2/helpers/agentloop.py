"""Drive a task's tools until it answers, or until it has had enough turns.

Every other task is one request and one reply. s15_shopping is an environment: the
model searches, sees results, searches again, and only then recommends. That needs
a loop. Ported from ``bench/surfaces/shared/agentloop.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

MAX_TURNS = 8


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments come back as a JSON *string*; a malformed one is empty."""
    raw = ((call.get("function") or {}).get("arguments")) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:                                       # noqa: BLE001
        return {}


def run_agent(adaptor, trial, surface, *, terminal: str, max_turns: int = MAX_TURNS,
              remind: str | None = None):
    """Run the tool loop. Returns ``(calls, transcript, error)``.

    Stops on ``terminal``, on a second reply with no tool calls, or at
    ``max_turns``. ``remind`` is sent once if the model answers in prose without
    having called the terminal tool.
    """
    messages = [dict(m) for m in trial.conversation.messages]
    calls: list[dict[str, Any]] = []
    nudged = False

    for _ in range(max_turns):
        step = replace(trial, conversation=replace(trial.conversation, messages=list(messages)))
        resp = adaptor.run(step)
        if resp.error:
            return calls, messages, resp.error

        raw_calls = (resp.usage or {}).get("raw_tool_calls") or []
        if not raw_calls:
            messages.append({"role": "assistant",
                             "content": [{"type": "text", "text": resp.text or ""}]})
            if nudged or not remind:
                return calls, messages, None
            nudged = True
            messages.append({"role": "user", "content": [{"type": "text", "text": remind}]})
            continue

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
