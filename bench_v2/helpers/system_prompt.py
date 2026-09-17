"""The old home of the scheme builders. Now a compatibility surface, and nothing more.

Everything moved:

* the **structural parts** -- the message constructors, the memory tree, the scripted
  ``list_dir``/``view_image`` chain, the tool definitions -- to
  :mod:`bench_v2.helpers.prompt_parts`;
* the **wording** -- every string a scheme sends -- to one ``.j2`` component per
  piece, in :mod:`bench_v2.helpers.schemes.components`, where a task overrides one by
  dropping a file into its own ``<scheme>/`` directory;
* the **shapes** -- one package per scheme -- to :mod:`bench_v2.helpers.schemes`.

This module used to hold all three schemes, their wording and their tool chain in one
300-line file. It survives only because other modules and tests already import these
names; new code should import from ``prompt_parts``, ``schemes`` or
``schemes.components`` directly.
"""

from __future__ import annotations

from bench_v2.helpers.prompt_parts import (  # noqa: F401
    HOMETOWN_SHARE,
    ME_DIR,
    ME_FILE,
    MEMORY_DIRS,
    PERSONA_VARIANTS,
    TOOLS,
    files_by_dir,
    memory_files,
)
from bench_v2.helpers.schemes import (  # noqa: F401
    BUILDERS,
    build_scheme_messages,
    builder_for,
)
from bench_v2.helpers.schemes.components import (  # noqa: F401
    CHAT_ACK,
    COMPONENTS,
    DEFAULT_AGENTIC_LIVE_SYSTEM,
    DEFAULT_AGENTIC_SYSTEM,
    DEFAULT_CHAT_SHARE,
    DEFAULT_LIVE_INTENT,
    LEGACY_NEWS_DIGEST_STYLE,
    LOOKED_ACK,
    ME_SENTENCE,
    MEMORY_CLAUSE_AGENTIC,
    MEMORY_CLAUSE_CHAT,
    Prompts,
    style_of,
)

# Computed once, as before, for anything that read them off the module.
FILES_BY_DIR = files_by_dir(3)
FILENAMES = [f for _dir, names in FILES_BY_DIR for f in names]
FILENAMES_LINE = "  ".join(FILENAMES)

# The old private name, kept for callers that reached for it.
_style_of = style_of

__all__ = [
    "PERSONA_VARIANTS", "MEMORY_DIRS", "ME_DIR", "ME_FILE", "ME_SENTENCE",
    "HOMETOWN_SHARE", "FILES_BY_DIR", "FILENAMES", "FILENAMES_LINE", "TOOLS",
    "DEFAULT_CHAT_SHARE", "DEFAULT_AGENTIC_SYSTEM", "DEFAULT_AGENTIC_LIVE_SYSTEM",
    "DEFAULT_LIVE_INTENT", "LEGACY_NEWS_DIGEST_STYLE", "CHAT_ACK", "LOOKED_ACK",
    "MEMORY_CLAUSE_CHAT", "MEMORY_CLAUSE_AGENTIC", "COMPONENTS", "Prompts",
    "BUILDERS", "build_scheme_messages", "builder_for",
    "files_by_dir", "memory_files", "style_of",
]
