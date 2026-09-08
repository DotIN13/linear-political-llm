"""Compatibility shim. s7_family_chat moved to ``bench/surfaces/tasks/s7_family_chat``.

Two test modules and ``bench/registry.py`` imported this path. Re-exporting is cheaper
than repointing them inside the same change that moves the code, and it is the same
call already made for ``generation.py``. **Temporary; do not add to it.**

``QUESTION_TEMPLATE`` is gone rather than re-exported. It was a Python format string
and it is a jinja file now, so its source says ``{{ message }}`` where the old name said
``{message}``: re-exporting it would leave ``.format(message=...)`` silently returning
the template unchanged. ``render_message()`` replaces it, and three call sites were
updated -- one pilot and two tests.
"""

from __future__ import annotations

from bench.surfaces.tasks.s7_family_chat import (  # noqa: F401
    DATASET_PATH, DOMAINS, FamilyChatSurface, dataset_fingerprint, load_dataset,
    question_template_source, render_message,
)
