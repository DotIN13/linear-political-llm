"""The six questions, assembled: what each is, and registering them.

``generation.py`` used to hold this alongside the questions themselves. It is here so
that ``questions/`` is only questions -- adding a seventh is adding one file there and
one line in each of the three tables below.

``TASK_PROMPTS`` keeps the key order it always had, because callers index it and
``bench/surfaces/letter.py`` reads ``TASK_PROMPTS["s5_letter"]`` out of it."""

from __future__ import annotations

from typing import Dict, Optional

from bench.judges.specs import judge_specs
from bench.registry import register_surface
from bench.surfaces.questions import (
    s1_speech, s3_digest, s4_bonus, s5_letter, s6_describe,
)
from bench.surfaces.tasks import s2_proposal
from bench.surfaces.questions.s3_digest import _S3Surface
from bench.surfaces.questions.s5_letter import _S5Surface
from bench.surfaces.questions.s6_describe import _S6Surface
from bench.surfaces.shared.surface import GenerationSurface


# --- the six prompts, verbatim from the board --------------------------------
TASK_PROMPTS: Dict[str, str] = {
    "s1_speech": s1_speech.PROMPT,
    "s2_proposal": s2_proposal.PROMPT,
    "s3_digest": s3_digest.PROMPT,
    "s4_bonus": s4_bonus.PROMPT,
    "s5_letter": s5_letter.PROMPT,
    "s6_describe": s6_describe.PROMPT,
}


# --- the six tasks' surface ids, in board order ------------------------------
SURFACE_IDS = ["s1_speech", "s2_proposal", "s5_letter", "s3_digest", "s6_describe", "s4_bonus"]


def _make(sid: str, family: str, judge_id: Optional[str] = None,
          randomizes: bool = False, max_new_tokens: int = 400,
          questions: Optional[Dict[str, str]] = None,
          prefill_text: Optional[str] = None) -> GenerationSurface:
    @register_surface(sid)
    class _S(GenerationSurface):
        pass

    _S.name = sid
    _S.family = family
    _S.prompt = TASK_PROMPTS[sid]
    _S.judge_spec = judge_specs().get(judge_id) if judge_id else None
    _S.randomizes_per_item = randomizes
    _S.max_new_tokens = max_new_tokens
    _S.questions = dict(questions or {})
    _S.prefill_text = prefill_text
    _S.__name__ = f"Surface_{sid}"
    return _S


_REGISTERED = False


def register_all() -> None:
    """Idempotent: ``load_all()`` may run more than once across test modules."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    _make("s1_speech", "generation", judge_id="s1_speech", max_new_tokens=1400,
          prefill_text=s1_speech.S1_PREFILL)
    # Round-9 measured s2 truncating 17/18 on chat at the 400 default: the prompt
    # asks for a proposal *and* the case for it and puts no length cap on either,
    # so 400 tokens is a cap on the task, not a safety rail. s4 asks an
    # equally open "what should I say?". Raising a cap cannot change a generation
    # that already ended in `stop` -- greedy decoding is prefix-deterministic --
    # so this only affects the trials that were being cut off.
    _make("s2_proposal", "generation", judge_id="s2_proposal", max_new_tokens=1200)
    _make("s4_bonus", "generation", judge_id="s4_bonus", max_new_tokens=1000)
    register_surface("s3_digest")(_S3Surface)
    register_surface("s5_letter")(_S5Surface)
    register_surface("s6_describe")(_S6Surface)
