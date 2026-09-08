"""Compatibility shim. s8_letter_answered moved to
``bench/surfaces/tasks/s8_letter_answered``.

One test module and ``bench/registry.py`` imported this path. Re-exporting is cheaper
than repointing them inside the same change that moves the code -- the same call
already made for ``generation.py`` and ``groupchat.py``. **Temporary; do not add.**
"""

from __future__ import annotations

from bench.surfaces.tasks.s8_letter_answered import (  # noqa: F401
    ASSISTANT_ASKS, DATASET_PATH, DOMAINS, OPENING_ASK, AnsweredLetterSurface,
    dataset_fingerprint, load_dataset,
)
