"""The few statistics a benchmark summary needs, stdlib only (no scipy on midway)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def mean_se(values: Sequence[float]) -> Tuple[float, float, int]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n), n


def ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, ties shared."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 3 or n != len(y):
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(ranks(x), ranks(y))


def corr_se(r: Optional[float], n: int) -> Optional[float]:
    """Large-sample standard error of a correlation, via Fisher's z."""
    if r is None or n < 4 or abs(r) >= 1:
        return None
    return (1 - r * r) / math.sqrt(n - 3)


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """P(a random positive scores above a random negative), ties count half."""
    if not pos or not neg:
        return None
    r = ranks(list(pos) + list(neg))
    rank_sum = sum(r[: len(pos)])
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))
