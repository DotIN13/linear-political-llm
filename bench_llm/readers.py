"""Helpers for reading a reply. A pilot picks the ones its answer format needs.

* ``option_probs`` -- the first generated token's probabilities over single-token
  options (scale digits, choice letters), renormalised, with the raw ``mass`` they
  held, so a reply that did not start with an option stays visible.
* ``expected_value`` -- the probability-weighted value of the options, or None
  below ``min_mass``.
* ``log_odds`` -- ln P(group a) / P(group b). Renormalising is a softmax over the
  options' logits, so logit gaps survive it; this keeps separating replies that an
  expected value has already pinned near 0.99. None when a group is absent.
* ``first_mention`` -- which of several keyword groups a free-text reply names
  first, within its head.
* ``matches`` -- whether a regex hits the head of a reply (a refusal, a "neither").

Every reader returns None rather than a guess.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


def clean_token(token: str) -> str:
    return token.strip().strip('*"\'`.:()[]').strip()


def option_probs(logprobs: Optional[Mapping[str, float]],
                 options: Sequence[str]) -> Tuple[Dict[str, float], float]:
    raw = {o: 0.0 for o in options}
    for token, lp in (logprobs or {}).items():
        t = clean_token(token)
        if t in raw:
            raw[t] += math.exp(lp)
    mass = sum(raw.values())
    if mass <= 0:
        return {}, 0.0
    return {k: v / mass for k, v in raw.items()}, mass


def top_option(probs: Mapping[str, float]) -> Tuple[Optional[str], Optional[float]]:
    if not probs:
        return None, None
    k = max(probs, key=probs.get)
    return k, probs[k]


def expected_value(probs: Mapping[str, float], values: Mapping[str, float], mass: float,
                   min_mass: float = 0.5) -> Optional[float]:
    if not probs or mass < min_mass:
        return None
    return sum(p * values[k] for k, p in probs.items())


def log_odds(probs: Mapping[str, float], a: Iterable[str], b: Iterable[str]) -> Optional[float]:
    pa = sum(probs.get(k, 0.0) for k in a)
    pb = sum(probs.get(k, 0.0) for k in b)
    if pa <= 0 or pb <= 0:
        return None
    return math.log(pa / pb)


def first_mention(text: str, groups: Mapping[str, Sequence[str]], window: int = 400) -> Optional[str]:
    """The name of the group whose keyword appears first in ``text[:window]``."""
    head = (text or "")[:window].lower()
    first: Dict[str, int] = {}
    for name, words in groups.items():
        hits = [m.start() for w in words for m in re.finditer(r"\b" + re.escape(w.lower()), head)]
        if hits:
            first[name] = min(hits)
    return min(first, key=first.get) if first else None


def matches(text: str, pattern: str | re.Pattern, window: int = 400) -> bool:
    rx = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
    return bool(rx.search((text or "")[:window]))


def word_count(text: str) -> int:
    return len((text or "").split())
