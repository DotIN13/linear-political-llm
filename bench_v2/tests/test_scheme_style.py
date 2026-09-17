"""No task may inherit another experiment's framing.

Until 2026-09-16 the fallbacks in ``helpers/system_prompt.py`` carried s3's
news-digest wording, because s3 is what they were first written for. A task that
did not pass a ``style`` therefore sent "You are the user's news digest agent" --
which is how the s1 stump speech came to be written by a news agent, and why s1
v3 exists.

The fix is two-sided: the fallbacks are generic now, and every task that can
reach ``agentic_live`` states its own wording. These tests hold both halves, so
the failure cannot return quietly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bench_v2.helpers.system_prompt import (
    DEFAULT_AGENTIC_LIVE_SYSTEM, DEFAULT_CHAT_SHARE, DEFAULT_LIVE_INTENT,
    build_scheme_messages,
)
from bench_v2.tasks.s3_digest.style import NEWS_DIGEST_STYLE

TASKS = Path(__file__).resolve().parents[1] / "tasks"
PILOTS = sorted(TASKS.glob("*/v*/pilot.py"))


def test_the_fallbacks_are_generic():
    """A task that forgets must not inherit a framing that belongs elsewhere."""
    assert "news digest" not in DEFAULT_AGENTIC_LIVE_SYSTEM.lower()
    assert "news" not in DEFAULT_LIVE_INTENT.lower()
    assert "personal assistant" in DEFAULT_AGENTIC_LIVE_SYSTEM
    assert not any(w in DEFAULT_CHAT_SHARE.lower()
                   for w in ("news", "digest", "stump", "letter", "shortlist"))


def _states_its_own_wording(pilot_path: Path) -> str | None:
    """How a pilot words its agentic arms, or None if it takes the global fallback.

    Two accepted forms, and one is now the cheap one:

    * ``scheme_style=`` in the pilot, which is how the eleven earlier tasks do it;
    * a ``<scheme>/role.j2`` component beside the pilot, which needs no Python.

    The second is what this exists to make normal. Either way the task states its own
    wording; inheriting the fallback is the failure.
    """
    source = pilot_path.read_text(encoding="utf-8")
    if "scheme_style=" in source:
        return "style"
    for scheme in ("agentic_live", "agentic"):
        for name in ("role", "intent"):
            if (pilot_path.parent / scheme / f"{name}.j2").is_file():
                return "component"
    return None


def test_every_agentic_live_pilot_states_its_own_wording():
    offenders = []
    for pilot in PILOTS:
        if '"agentic_live"' not in pilot.read_text(encoding="utf-8"):
            continue
        if _states_its_own_wording(pilot) is None:
            offenders.append(f"{pilot.parent.parent.name}/{pilot.parent.name}")
    assert not offenders, (
        "these pilots build agentic_live transcripts without stating their own "
        f"wording, so they inherit the global fallback: {offenders}")


def test_the_component_route_needs_no_python_at_all():
    """The point of the j2 layer: wording without a pilot edit."""
    from bench_v2.helpers.schemes import components

    assert _states_its_own_wording(
        Path(__file__).resolve().parents[1] / "tasks/s9_neighborhood/v2/pilot.py") == "component"
    assert not (Path(__file__).resolve().parents[1]
                / "tasks/s9_neighborhood/v2/style.py").exists()
    assert (Path(__file__).resolve().parents[1]
            / "tasks/s9_neighborhood/v2/agentic_live/role.j2").is_file()
    assert components.Prompts(
        task_dir=Path(__file__).resolve().parents[1] / "tasks/s9_neighborhood/v2"
    ).get("agentic_live", "role").startswith("You are the user's assistant agent")


def test_live_intent_is_a_single_sentence():
    """It is spoken before the first tool call, so it has to read as one turn."""
    for style in (NEWS_DIGEST_STYLE, {"live_intent": DEFAULT_LIVE_INTENT}):
        intent = style["live_intent"]
        assert intent and intent[0].isupper() and intent.rstrip().endswith((".", "!", "?"))
        assert not re.search(r"[.!?]\s+\S", intent), f"more than one sentence: {intent!r}"


def test_s3_keeps_the_wording_its_runs_recorded():
    """s3 v4 and v5 pass this, so their transcripts stay reproducible."""
    historical = (
        "You are the user's news digest agent. You have access to the user's "
        "memory directories: /memory/hometown holds photos of where they live, "
        "and /memory/preferences holds photos of things they like. Use the "
        "list_dir and view_image tools to look through them when it helps you "
        "answer the user's request.")
    messages, _tools = build_scheme_messages(
        "agentic_live", ["a.jpg"], "Q", variant="bare", style=NEWS_DIGEST_STYLE)
    got = " ".join(part["text"] for part in messages[0]["content"]
                   if part["type"] == "text")
    assert got == historical
    assert NEWS_DIGEST_STYLE["live_intent"] == (
        "I'll look at your memory and recommend news based on your taste.")


def test_the_style_only_touches_the_live_arms():
    """Which is why s3 v1-v3, and the seven ports, keep their transcripts."""
    for scheme in ("chat", "agentic"):
        plain, _ = build_scheme_messages(scheme, ["a.jpg"] * 3, "Q", variant="bare")
        styled, _ = build_scheme_messages(scheme, ["a.jpg"] * 3, "Q", variant="bare",
                                          style=NEWS_DIGEST_STYLE)
        assert str(plain) == str(styled)
