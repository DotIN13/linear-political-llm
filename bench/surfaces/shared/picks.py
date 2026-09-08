"""Readers for the pick-from-a-pool surfaces, and the question body they share.

Five of the six new tasks ask the model to choose from a numbered list and answer
with numbers; the sixth asks it to call one of eight tools. This module holds the
three readers that turn those answers back into indices, and the one function that
renders the list.

**Every reader returns ``None`` rather than a guess.** A record that cannot be read
is kept and excluded from the dependent variable, which is the only way a parse
failure stays visible instead of quietly becoming a number. The last ranking task we
ran parsed at 79.5%, and the missing fifth was not missing at random -- so the
difference between "unreadable" and "zero" is the difference between knowing that
and not.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

# "1, 3, 5" possibly behind a short lead-in ("I'd pick 1, 3 and 5"). The 12-character
# budget is deliberate: it admits a few words of preamble and rejects a sentence.
_NUM_LINE = re.compile(r"^[^0-9]{0,12}((?:\d{1,2})(?:\s*[,、和and]+\s*\d{1,2})*)", re.I)

FORMAT_PICK_N = ("\n\nStart your reply with just the {n} numbers, comma-separated, "
                 "on a line of their own. Then give your reasons.")
FORMAT_RANK_N = ("\n\nStart your reply with all {n} numbers in your order, best first, "
                 "comma-separated, on a line of their own. Then give your reasons.")


def numbered_question(ask: str, options: Sequence[Dict[str, Any]],
                      render: Callable[[Dict[str, Any]], str], suffix: str = "") -> str:
    """``ask``, a blank line, the options numbered from 1, then the format line.

    The options arrive **already in presentation order** -- ordering is the caller's
    job, because it is an experimental factor and not a rendering detail.
    """
    lines = [f"{i + 1}. {render(row)}" for i, row in enumerate(options)]
    return ask + "\n\n" + "\n".join(lines) + suffix


def _leading_numbers(text: str, n_options: int) -> Optional[List[int]]:
    """The first line that begins with a run of in-range integers, de-duplicated."""
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


def parse_picks(text: str, n_expected: int, n_options: int) -> Optional[List[int]]:
    """Exactly ``n_expected`` distinct 1-based picks, or ``None``.

    Strict on the count in both directions. Two numbers when three were asked for is
    not two picks -- it is an answer to a different question, and averaging it in
    would quietly reweight the trial.
    """
    nums = _leading_numbers(text, n_options)
    if nums is None or len(nums) != n_expected:
        return None
    return nums


def parse_ranking(text: str, n_options: int) -> Optional[List[int]]:
    """A full permutation of ``1..n_options``, or ``None``.

    A partial ranking is not a ranking: the dependent variable is the *position* of
    one option, and a list that omits it has no position to report.
    """
    nums = _leading_numbers(text, n_options)
    if nums is None or sorted(nums) != list(range(1, n_options + 1)):
        return None
    return nums


def parse_tool_call(text: str, tool_names: Sequence[str]) -> Optional[str]:
    """The **first** of ``tool_names`` to appear in the generated text.

    A tool call surfaces in the transcript as the function's name, so the reading is
    a search rather than a parse -- and the first occurrence is the answer even when
    the model goes on to discuss the others, which is what "don't split it across
    two" in the ask is there to discourage.

    Longest name first, so no name that is a prefix of another can shadow it.
    """
    body = text or ""
    best: Optional[str] = None
    best_at = len(body) + 1
    for name in sorted(tool_names, key=len, reverse=True):
        at = body.find(name)
        if at != -1 and at < best_at:
            best, best_at = name, at
    return best


def rank_weights(n: int) -> List[float]:
    """Rank-correlation weights: +1 for first, -1 for last, evenly spaced.

    ``w(r) = (n + 1 - 2r) / (n - 1)``. Sums to zero, so a random ranking scores 0 in
    expectation and a positive score means the coded pole was ranked earlier. For
    ``n = 6`` this is the (7 - 2r)/5 given in s11's meta.
    """
    if n < 2:
        raise ValueError("rank weights need at least two options")
    return [(n + 1 - 2 * r) / (n - 1) for r in range(1, n + 1)]


def picked_mean(rows: Sequence[Dict[str, Any]], picks: Sequence[int], field: str) -> float:
    """Mean of ``field`` over the picked rows. ``picks`` are 1-based into ``rows``."""
    return sum(float(rows[i - 1][field]) for i in picks) / len(picks)


def shown_mean(rows: Sequence[Dict[str, Any]], field: str) -> float:
    return sum(float(r[field]) for r in rows) / len(rows)
