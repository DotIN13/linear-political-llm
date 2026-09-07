"""Text normalisation the readers share.

``_normalize_apostrophes`` is here rather than in ``refusal.py`` because two readers
need it: the refusal scan and s3's token matcher. ``word_count`` is on every
generation outcome, so it is shared by every question rather than owned by one."""

from __future__ import annotations


def _normalize_apostrophes(text: str) -> str:
    """Curly quotes the model emits (U+2018/U+2019) count as ASCII apostrophes."""
    return (text or "").replace("\u2019", "'").replace("\u2018", "'")


def word_count(text: str) -> int:
    return len((text or "").split())
