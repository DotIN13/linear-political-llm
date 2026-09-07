"""Outlet names: normalising them, and finding them in an answer.

Shared per the layout. Only s3 reads an outlet name today, but the four
forced-choice surfaces now being designed may well, and an outlet name is a
general enough thing that sharing it is a reasonable bet rather than a claim."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple


_OUTLET_SUFFIX = re.compile(r"\s*\((?:website|online|opinion)\)\s*$", re.I)


def normalize_outlet(name: str) -> str:
    """``"Fox News (website)"`` -> ``"fox news"``. The suffix is Ad Fontes', not the
    outlet's own name, and the model never writes it."""
    return re.sub(r"\s+", " ", _OUTLET_SUFFIX.sub("", name or "")).strip().lower()


def outlet_matches(segment: str, headlines: Sequence[Dict[str, Any]]) -> List[int]:
    """Headline indices whose outlet name appears verbatim in ``segment``.

    Outlet names are reproduced verbatim by the model even when it paraphrases the
    headline (docs/bench/13 §2), and they are unique within the stimulus set -- so
    this is a deterministic signal, not a guess. Longest name wins on nesting
    (``"Fox Business"`` beats ``"Fox"``); a genuinely ambiguous segment returns
    every match and the caller declines to use it.
    """
    seg = re.sub(r"\s+", " ", (segment or "")).lower()
    found: List[Tuple[int, int]] = []           # (length, index)
    for i, h in enumerate(headlines):
        name = normalize_outlet(h.get("outlet", ""))
        if not name:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", seg):
            found.append((len(name), i))
    if not found:
        return []
    longest = max(n for n, _ in found)
    # Drop names that are a substring of a longer match in the same segment.
    keep = [i for n, i in found
            if not any(n2 > n and normalize_outlet(headlines[i].get("outlet", ""))
                       in normalize_outlet(headlines[j].get("outlet", ""))
                       for n2, j in found)]
    return sorted(keep) if keep else sorted(i for n, i in found if n == longest)
