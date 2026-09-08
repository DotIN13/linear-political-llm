"""s12_explain: which three points would you cover for a twelve-year-old.

Two questions in one surface, and that is the whole design:

  ``inflation``  a live topic -- four points blaming corporate pricing and market
                 power, four blaming government spending and the money supply
  ``sky_blue``   an inert topic -- eight true statements about Rayleigh scattering,
                 coded on an axis of the same shape with no side to take

**The control is inside the task rather than beside it.** Same ask, same length, same
option count, same parser; only the valence differs. If the photos move the inflation
picks and not the sky-blue picks, that is the effect. If they move both, the model is
being shifted generally and the inflation reading means much less.

DV ``code_pick_rel`` = mean ``code_c`` over the three picked, minus the mean over the
eight shown. All eight are always shown so the second term is exactly 0 -- it is
written in relative form anyway, so a sensitivity analysis that drops a point stays
correct. That is s3's ``slant_rel_mean`` lesson, applied before it can bite.

Anchor: Navigator Research's own two named causes, with published marginals --
D 72 / R 45 on one and D 42 / R 78 on the other.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bench.surfaces.shared.picks import (FORMAT_PICK_N, numbered_question, parse_picks,
                                         picked_mean, shown_mean)
from bench.surfaces.shared.prompts import pool, text
from bench.surfaces.shared.surface import GenerationSurface
from bench.types import Trial

N_PICKS = 3
TOPICS = ["inflation", "sky_blue"]
ASKS = {t: text(__file__, f"ask_{t}.txt") for t in TOPICS}
PROMPT = ASKS["inflation"]


def load_points() -> List[Dict[str, Any]]:
    rows, _header = pool(__file__, "points")
    return rows


class _S12Surface(GenerationSurface):
    name = "s12_explain"
    family = "pick_from_pool"
    max_new_tokens = 700
    prompt = PROMPT
    questions = dict(ASKS)

    def __init__(self) -> None:
        rows = load_points()
        self.by_topic = {t: [r for r in rows if r["topic"] == t] for t in TOPICS}



    def question(self, order: Optional[List[int]] = None, attribution: str = "shown",
                 qid: Optional[str] = None) -> str:
        qid = qid or TOPICS[0]
        shown = self._shown(qid, order)
        return numbered_question(ASKS[qid], shown, lambda r: r["text"],
                                 FORMAT_PICK_N.format(n=N_PICKS))

    def _shown(self, qid: str, order: Optional[List[int]]) -> List[Dict[str, Any]]:
        rows = self.by_topic[qid]
        return list(rows) if order is None else [rows[i] for i in order]

    def _deterministic(self, text_out: str, trial: Optional[Trial]) -> Dict[str, Any]:
        variant = (trial.variant if trial else {}) or {}
        qid = str(variant.get("question", TOPICS[0]))
        order = list(variant["order"]) if variant.get("order") else None
        shown = self._shown(qid, order)
        picks = parse_picks(text_out, N_PICKS, len(shown))
        if picks is None:
            return {"parsed": False, "picks": None, "topic": qid}
        rel = picked_mean(shown, picks, "code_c") - shown_mean(shown, "code_c")
        return {
            "parsed": True,
            "topic": qid,
            "picks": picks,
            "picked_ids": [shown[i - 1]["pid"] for i in picks],
            "primary": rel,
            "code_pick_rel": rel,
            "code_pick_mean": picked_mean(shown, picks, "code_c"),
            "right_pick_mean": picked_mean(shown, picks, "right_c"),
        }
