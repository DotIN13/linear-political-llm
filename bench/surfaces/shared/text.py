"""Text normalisation the readers share.

``_normalize_apostrophes`` is here rather than in ``refusal.py`` because two readers
need it: the refusal scan and s3's token matcher. ``word_count`` is on every
generation outcome, so it is shared by every task rather than owned by one.

``_norm_tokens`` and ``token_set_similarity`` are here because the layout puts them
here. Note that s3's calibration -- the 0.70 coverage threshold and the 0.10
ambiguity margin -- deliberately stayed in ``tasks/s3_digest``: the function is
general, the numbers are not."""

from __future__ import annotations

import re
from typing import List


def _normalize_apostrophes(text: str) -> str:
    """Curly quotes the model emits (U+2018/U+2019) count as ASCII apostrophes."""
    return (text or "").replace("\u2019", "'").replace("\u2018", "'")


def word_count(text: str) -> int:
    return len((text or "").split())


def _norm_tokens(text: str) -> List[str]:
    t = _normalize_apostrophes(text or "").lower()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return t.split()


def token_set_similarity(segment: str, headline: str) -> float:
    """Token-set *coverage*: share of the headline's tokens present in the segment.

    Coverage (not Jaccard) is the right metric here because a picked headline is
    usually quoted verbatim and then followed by a sentence of its own -- the
    extra sentence words must not dilute the score. 1.0 == every headline token
    appears in the segment.
    """
    ht = set(_norm_tokens(headline))
    if not ht:
        return 0.0
    return len(ht & set(_norm_tokens(segment))) / len(ht)
