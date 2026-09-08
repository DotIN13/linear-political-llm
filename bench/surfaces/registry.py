"""The eight tasks, assembled: what each is, and registering them.

``generation.py`` used to hold this alongside the questions themselves. It is here so
that ``tasks/`` is only tasks -- adding a ninth is adding one directory there and one
line in each of the three tables below.

All eight register in ``register_all()``. s7 and s8 used to register themselves with a
decorator at class-definition time, which made them the only two registered anywhere
else; now there is one place and one moment.

``TASK_PROMPTS`` keeps the key order it always had, because callers index it. Note it
holds **six** entries, not eight: s7 and s8 have twelve questions each rather than one
prompt, and ``SURFACE_IDS`` likewise names the six -- ``bench/pilots/round9_vllm.py``
iterates it, and quietly making it eight would have that pilot start running two tasks
it was never asked to. Both names could stand to say "generation" in them; that is a
rename, not this pass."""

from __future__ import annotations

from typing import Dict, Optional

from bench.judges.specs import judge_specs
from bench.registry import register_surface
from bench.surfaces.tasks import (
    s1_speech, s2_proposal, s3_digest, s4_bonus, s5_letter, s6_describe,
)
from bench.surfaces.tasks.s3_digest import _S3Surface
from bench.surfaces.tasks.s7_family_chat import FamilyChatSurface
from bench.surfaces.tasks.s8_letter_answered import AnsweredLetterSurface
from bench.surfaces.tasks.s5_letter import _S5Surface
from bench.surfaces.tasks.s9_neighborhood import _S9Surface
from bench.surfaces.tasks.s10_groceries import _S10Surface
from bench.surfaces.tasks.s11_health import _S11Surface
from bench.surfaces.tasks.s12_explain import _S12Surface
from bench.surfaces.tasks.s14_outfits import _S14Surface
from bench.surfaces.tasks.s6_describe import _S6Surface
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

# The five forced-choice surfaces, in the order Tianyi ranked them. Kept as their own
# list rather than appended to SURFACE_IDS: that name is read by round9_vllm.py and by
# the prompt-golden test as "the six generation tasks", and widening it silently would
# change what those two mean.
CHOICE_SURFACE_IDS = ["s9_neighborhood", "s12_explain", "s11_health",
                      "s10_groceries", "s14_outfits"]


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
    # s7 and s8 used to register themselves with a decorator at import time, which
    # made them the only surfaces not registered here. Same six-then-two calls, one
    # place, one moment.
    register_surface("s7_family_chat")(FamilyChatSurface)
    register_surface("s8_letter_answered")(AnsweredLetterSurface)
    register_surface("s5_letter")(_S5Surface)
    register_surface("s6_describe")(_S6Surface)
    # The forced-choice five. Each reads its own option pool and its own DV; none has
    # a judge, because every one of them is read by rule.
    register_surface("s9_neighborhood")(_S9Surface)
    register_surface("s12_explain")(_S12Surface)
    register_surface("s11_health")(_S11Surface)
    register_surface("s10_groceries")(_S10Surface)
    register_surface("s14_outfits")(_S14Surface)
