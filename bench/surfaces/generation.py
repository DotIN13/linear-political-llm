"""GenerationSurface + the six open-ended tasks (docs/bench board-tasks).

One implementation class, six configs. Every surface is an open-ended prompt
with no political word in it; the politics must come out in the answer
(board-tasks: "提示词里不许出现政治词，政治必须出现在回答里").

Two delivery schemes are variants (board-tasks), not two surfaces:

* ``scheme="chat"``    -- the user hands over 3 photos and says "I took these".
* ``scheme="agentic"`` -- a memory agent lists a directory and reads the 3 photos
  through ``view_image`` tool calls (the validated shape from docs/bench/08).

The shared prefix -- everything before the final question -- is byte-identical
across all six surfaces within a scheme, so ``s_pre`` (the probe read at the end
of that prefix) is identical across surfaces *by construction*.

Engineering rules re-applied from docs/bench/08: every message ``content`` is a
list; the agentic system text is folded into the first user turn (Gemma); the
conversation carries ``tool_calls``/``role:"tool"`` messages and the tools list
rides in ``trial.meta["tools"]`` (``encode_prompts`` has no ``tools=`` entry).

s3_digest is the only surface that needs item-specific material: twelve
headlines and their Ad Fontes slant, read from ``bench/data/s3_headlines_v2.json``
and re-ordered (deterministically, seeded by ``(item_id, seed)``) every trial.
That order goes into ``variant["order"]`` so two orders never collide on one
``trial_key``. The headline rows are rendered with or without their outlet name
(``variant["attribution"]``, default ``"shown"``).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from bench.judges.specs import judge_specs
from bench.registry import register_surface
from bench.types import (
    Capability, Conversation, Item, Outcome, ProbePoint, Response, Trial,
)
from bench.surfaces.questions import s1_speech, s2_proposal, s3_digest, s4_bonus, s5_letter, s6_describe
from bench.surfaces.shared.text import (
    _normalize_apostrophes, word_count,
)
from bench.surfaces.shared.refusal import (
    REFUSAL_WINDOW, _REFUSAL_PATTERNS, _refusal_match, detect_refusal,
)
from bench.surfaces.shared.conditions import (
    CONDITIONS, CONDITION_ALIASES, CONDITION_DESC, REMOVED_CONDITIONS, normalise_condition,
)
from bench.surfaces.shared.ordering import (
    _order_seed, sampled_order, shuffled_order,
)
from bench.surfaces.shared.transcript import (
    AGENTIC_ACK, AGENTIC_OPENER, ASSISTANT_TURN_1, ASSISTANT_TURN_2, CHAT_USER_TURN_2, FILENAMES, FILENAMES_LINE, FILES_BY_DIR, HOMETOWN_SHARE, MEMORY_DIRS, SHARE_LINE, SYSTEM_AGENTIC, TOOLS, _FILENAME_POOL, _agentic_messages, _chat_messages, _tool_call, build_scheme_messages, files_by_dir,
)
from bench.surfaces.shared.surface import (
    GenerationSurface,
)
from bench.surfaces.questions.s3_digest import (
    ROOT_DIR, S3_AMBIGUITY_MARGIN, S3_HEADLINES_PATH, S3_MATCH_THRESHOLD, S3_N_PICKS, _OUTLET_SUFFIX, _S3Surface, _find_index_markers, _norm_tokens, _split_segments, extract_picks, load_s3_headlines, normalize_outlet, outlet_matches, token_set_similarity,
)
from bench.surfaces.questions.s5_letter import (
    TOPIC_KEYWORDS, TOPIC_LEAN, _S5Surface, extract_topic,
)
from bench.surfaces.questions.s6_describe import (
    _POLITICS_WORDS, _S6Surface, extract_mentions_politics,
)
from bench.surfaces.questions.s1_speech import (
    S1_PREFILL,
)

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
          prefill_text=S1_PREFILL)
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
