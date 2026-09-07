"""One question, one file. Adding a seventh is adding a file here.

Each module owns its prompt (``PROMPT``), any reader only it uses, and its
``GenerationSurface`` subclass if it needs one. ``bench/surfaces/registry.py``
assembles them."""

from __future__ import annotations
