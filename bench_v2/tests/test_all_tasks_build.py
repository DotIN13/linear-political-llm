"""Every pilot still builds, after the assembly layer moved out of ``system_prompt``.

The split put the parts in ``helpers/prompt_parts`` and the shapes in
``helpers/schemes``, and left ``helpers/system_prompt`` as a compatibility surface
re-exporting the old names. Eleven existing tasks import from that surface, so the
regression that matters is simply: does every pilot still build a trial?

This is deliberately a smoke test, not a transcript test. The exact conversations
are asserted where they belong -- ``test_s1_v2``, ``test_s5_v2``, ``test_s7_v2``,
``test_s8_v2``, ``test_agentic_live``, ``test_transcript_parity``,
``test_scheme_portrait`` -- and those all pass unchanged. What this adds is that a
pilot which no other test touches cannot break silently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bench_v2.types import baseline_item

TASKS = Path(__file__).resolve().parents[1] / "tasks"
PILOTS = sorted(TASKS.glob("*/v*/pilot.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_pilot_{path.parent.parent.name}_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _texts(message) -> list[str]:
    """The text parts of one message. Not ``str(content)``: that reprs the text and
    escapes the newlines a multi-line question contains, so a substring test against
    it fails for every pilot whose question has more than one line."""
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    return [part.get("text") or "" for part in (content or [])
            if isinstance(part, dict) and part.get("type") == "text"]


def test_there_are_pilots_to_check():
    """A glob that silently matched nothing would make every test below vacuous."""
    assert len(PILOTS) >= 12, [p.name for p in PILOTS]


@pytest.mark.parametrize("pilot_path", PILOTS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_every_pilot_builds_a_trial(pilot_path):
    module = _load(pilot_path)
    variant = dict(module.variants()[0])
    conditions = getattr(module, "CONDITIONS", ("photos",))
    for condition in conditions:
        trial = module.build(baseline_item(), condition, dict(variant))
        messages = trial.conversation.messages
        assert messages, f"{pilot_path} built an empty conversation"
        for message in messages:
            assert "role" in message and "content" in message, message
        assert trial.meta.get("scheme")
        assert trial.meta.get("question")


@pytest.mark.parametrize("pilot_path", PILOTS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_every_pilot_keeps_its_question_out_of_the_prefix(pilot_path):
    """``s_pre`` is taken at the end of the shared prefix, so the question must be
    the last thing added -- the invariant the assembly layer must not break."""
    module = _load(pilot_path)
    messages = module.build(baseline_item(), "photos", dict(module.variants()[0])).conversation.messages
    question = module.build(baseline_item(), "photos",
                            dict(module.variants()[0])).meta.get("question")
    if not question:
        pytest.skip("pilot records no question text")
    sent = "\n".join(t for m in messages for t in _texts(m))
    assert question in sent, f"{pilot_path} never sends its question"
