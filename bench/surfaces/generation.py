"""Compatibility shim. The code moved; this keeps the import path.

``bench/surfaces/generation.py`` was 1,143 lines: six unrelated questions plus all
the machinery they share. It is now

    bench/surfaces/shared/      conditions, ordering, outlets, refusal, text,
                                transcript, surface
    bench/surfaces/questions/   one file per question
    bench/surfaces/registry.py  TASK_PROMPTS, SURFACE_IDS, _make, register_all

**This module re-exports all of it and defines nothing.** It exists because
``bench/pilots/*`` (eight files), ``bench/surfaces/letter.py``,
``bench/surfaces/groupchat.py``, ``bench/registry.py`` and six test modules import
from ``bench.surfaces.generation``, and repointing thirty-odd import sites in the
same change that moves the code would mean a diff nobody can review as
behaviour-preserving.

**It is temporary and it is not the place to add anything.** New code should import
from the real module. When the callers have been repointed -- a separate, mechanical
commit -- this file goes away.

Note for whoever does that: deleting it changes ``measurement_rev`` again, because
``MEASUREMENT_GLOBS`` hashes the path list as well as the contents
(``bench/store.py:25``).

The original module docstring follows verbatim, kept because it is the design record
for the six questions and the two delivery schemes.

------------------------------------------------------------------------------

GenerationSurface + the six open-ended tasks (docs/bench board-tasks).

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

from bench.surfaces.shared.text import (  # noqa: F401
    _norm_tokens, _normalize_apostrophes, token_set_similarity, word_count,
)
from bench.surfaces.shared.outlets import (  # noqa: F401
    _OUTLET_SUFFIX, normalize_outlet, outlet_matches,
)
from bench.surfaces.shared.refusal import (  # noqa: F401
    REFUSAL_WINDOW, _REFUSAL_PATTERNS, _refusal_match, detect_refusal,
)
from bench.surfaces.shared.conditions import (  # noqa: F401
    CONDITIONS, CONDITION_ALIASES, CONDITION_DESC, REMOVED_CONDITIONS,
    normalise_condition,
)
from bench.surfaces.shared.ordering import (  # noqa: F401
    _order_seed, sampled_order, shuffled_order,
)
from bench.surfaces.shared.transcript import (  # noqa: F401
    AGENTIC_ACK, AGENTIC_OPENER, ASSISTANT_TURN_1, ASSISTANT_TURN_2, CHAT_USER_TURN_2,
    FILENAMES, FILENAMES_LINE, FILES_BY_DIR, HOMETOWN_SHARE, MEMORY_DIRS, SHARE_LINE,
    SYSTEM_AGENTIC, TOOLS, _FILENAME_POOL, _agentic_messages, _chat_messages, _tool_call,
    build_scheme_messages, files_by_dir,
)
from bench.surfaces.shared.surface import (  # noqa: F401
    GenerationSurface,
)
from bench.surfaces.questions.s1_speech import (  # noqa: F401
    S1_PREFILL,
)
from bench.surfaces.questions.s3_digest import (  # noqa: F401
    ROOT_DIR, S3_AMBIGUITY_MARGIN, S3_HEADLINES_PATH, S3_MATCH_THRESHOLD, S3_N_PICKS,
    _S3Surface, _find_index_markers, _split_segments, extract_picks, load_s3_headlines,
)
from bench.surfaces.questions.s5_letter import (  # noqa: F401
    TOPIC_KEYWORDS, TOPIC_LEAN, _S5Surface, extract_topic,
)
from bench.surfaces.tasks.s6_describe import (  # noqa: F401
    _POLITICS_WORDS, _S6Surface, extract_mentions_politics,
)
from bench.surfaces.registry import (  # noqa: F401
    SURFACE_IDS, TASK_PROMPTS, register_all,
)
