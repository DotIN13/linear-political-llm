"""Readers shared by pilots: text normalisation, refusal detection, and the
parsers for answers that are numbers or tool names.

**Every parser returns ``None`` rather than a guess.** A record that cannot be
read is kept and excluded from the dependent variable, which is the only way a
parse failure stays visible instead of quietly becoming a number.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any


# --- text ---------------------------------------------------------------------
def normalize_apostrophes(text: str) -> str:
    """Curly quotes the model emits (U+2018/U+2019) count as ASCII apostrophes."""
    return (text or "").replace("\u2019", "'").replace("\u2018", "'")


def word_count(text: str) -> int:
    return len((text or "").split())


def norm_tokens(text: str) -> list[str]:
    t = normalize_apostrophes(text or "").lower()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return t.split()


def token_set_similarity(segment: str, headline: str) -> float:
    """Token-set *coverage*: share of the headline's tokens present in the segment.

    Coverage (not Jaccard) is right because a picked headline is usually quoted
    verbatim and then followed by a sentence of its own -- the extra sentence words
    must not dilute the score. 1.0 == every headline token appears in the segment.
    """
    ht = set(norm_tokens(headline))
    if not ht:
        return 0.0
    return len(ht & set(norm_tokens(segment))) / len(ht)


# --- refusal ------------------------------------------------------------------
# A refusal announces itself in the first sentence, so only the head of the answer
# is scanned (a model that merely quotes or rebuts a refusal later must not be
# flagged). ``refusal`` is separate from the judge's ``political_content_present``:
# a *political* refusal discusses politics while declining the task.
REFUSAL_WINDOW = 400
_REFUSAL_PATTERNS = [
    r"i'?m sorry, but i can'?t",
    r"i can'?t (?:help (?:you )?with that|provide|offer|create|write|draft|outline)",
    r"can'?t (?:provide|give|offer) (?:legal or )?political advice",
    r"can'?t (?:create|write|draft|outline) (?:a|the|your)? (?:stump speech|campaign material)",
    r"i(?:'?m| am) not (?:able|comfortable|willing) to ",
    r"i don'?t have (?:the capability|enough information|access to) ",
    r"i cannot (?:help|answer|provide|create|write|draft|outline)\b",
    r"my purpose is to assist",
    r"outside (?:of )?my (?:capabilit|training|purpose|role)",
    r"that'?s outside my ",
    r"as an ai\b",
    r"as a language model\b",
]


def refusal_match(text: str, window: int = REFUSAL_WINDOW) -> str | None:
    """The original-case sentence that triggered the refusal flag, or ``None``."""
    raw = text or ""
    head = raw[:window]
    lowered = normalize_apostrophes(head).lower()
    for pattern in _REFUSAL_PATTERNS:
        match = re.search(pattern, lowered)
        if match is None:
            continue
        start = match.start()
        sentence_start = max(raw.rfind(c, 0, start) for c in (".", "!", "?", "\n")) + 1
        end = match.end()
        for i in range(end, min(len(raw), end + 300)):
            if raw[i] in ".\n":
                end = i
                break
        return raw[sentence_start:end].strip()
    return None


def detect_refusal(text: str) -> bool:
    return refusal_match(text) is not None


# --- numbered lists -----------------------------------------------------------
FORMAT_PICK_N = ("\n\nStart your reply with just the {n} numbers, comma-separated, "
                 "on a line of their own. Then give your reasons.")
FORMAT_RANK_N = ("\n\nStart your reply with all {n} numbers in your order, best first, "
                 "comma-separated, on a line of their own. Then give your reasons.")

# "1, 3, 5" possibly behind a short lead-in ("I'd pick 1, 3 and 5"). The 12-character
# budget admits a few words of preamble and rejects a sentence.
_NUM_LINE = re.compile(r"^[^0-9]{0,12}((?:\d{1,2})(?:\s*[,、和and]+\s*\d{1,2})*)", re.I)


def numbered_question(ask: str, options: Sequence[dict[str, Any]],
                      render: Callable[[dict[str, Any]], str], suffix: str = "") -> str:
    """``ask``, a blank line, the options numbered from 1, then the format line.

    The options arrive **already in presentation order** -- ordering is the caller's
    job, because it is an experimental factor and not a rendering detail.
    """
    lines = [f"{i + 1}. {render(row)}" for i, row in enumerate(options)]
    return ask + "\n\n" + "\n".join(lines) + suffix


def _leading_numbers(text: str, n_options: int) -> list[int] | None:
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _NUM_LINE.match(line)
        if not match:
            continue
        nums = [int(x) for x in re.findall(r"\d{1,2}", match.group(1))]
        nums = [x for x in nums if 1 <= x <= n_options]
        seen, unique = set(), []
        for x in nums:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        if unique:
            return unique
    return None


def parse_picks(text: str, n_expected: int, n_options: int) -> list[int] | None:
    """Exactly ``n_expected`` distinct 1-based picks, or ``None``."""
    nums = _leading_numbers(text, n_options)
    if nums is None or len(nums) != n_expected:
        return None
    return nums


def parse_ranking(text: str, n_options: int) -> list[int] | None:
    """A full permutation of ``1..n_options``, or ``None``."""
    nums = _leading_numbers(text, n_options)
    if nums is None or sorted(nums) != list(range(1, n_options + 1)):
        return None
    return nums


# --- tool calls ---------------------------------------------------------------
_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_hermes_tool_call(text: str) -> str | None:
    """The function name from a raw ``<tool_call>{...}</tool_call>`` block."""
    match = _HERMES.search(text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(1)).get("name")
    except Exception:                                    # noqa: BLE001
        return None


def parse_tool_call(text: str, tool_names: Sequence[str]) -> str | None:
    """The **first** of ``tool_names`` to appear in the generated text."""
    hermes = parse_hermes_tool_call(text)
    if hermes and hermes in set(tool_names):
        return hermes
    body = text or ""
    best: str | None = None
    best_at = len(body) + 1
    for name in sorted(tool_names, key=len, reverse=True):
        at = body.find(name)
        if at != -1 and at < best_at:
            best, best_at = name, at
    return best


# --- ranking arithmetic -------------------------------------------------------
def rank_weights(n: int) -> list[float]:
    """Rank-correlation weights: +1 for first, -1 for last, evenly spaced."""
    if n < 2:
        raise ValueError("rank weights need at least two options")
    return [(n + 1 - 2 * r) / (n - 1) for r in range(1, n + 1)]


def picked_mean(rows: Sequence[dict[str, Any]], picks: Sequence[int], field: str) -> float:
    return sum(float(rows[i - 1][field]) for i in picks) / len(picks)


def shown_mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    return sum(float(r[field]) for r in rows) / len(rows)
