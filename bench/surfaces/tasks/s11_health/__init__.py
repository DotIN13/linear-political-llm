"""s11_health: put six ways of dealing with a symptom in the order you'd try them.

The one task that asks for a **full ranking** rather than a shortlist, and the only
one whose DV is a position rather than a mean over picks.

Two scenarios, ``sleep`` and ``allergies``, each with the same six routes on one
institutional axis: doctor and prescription at the credentialled pole, pharmacist and
self-management in the middle, supermarket supplement and herbalist at the other.

DV ``rank_of_clinician`` -- the 1..6 position of the ``clinician`` route. One number,
reported on its own. **LOWER means sent to a doctor sooner.**
Secondary ``inst_rank_score`` = sum of ``institution_c(i) * w(rank_i)`` with
``w(r) = (7 - 2r)/5``: 0 in expectation for a random ranking, positive when
credentialled routes come earlier.

Anchor: Pew trust in scientists -- 88% against 66% for confidence, 80% against 52%
for honesty. ``right_c = -institution_c``, because Pew puts Democrats higher.

**Expect a ceiling.** The ranking board's own warning is that this will very likely
put the doctor first every time, which would make the mean uninformative and the
*variance across personas* the only thing worth reading. Report the variance first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bench.surfaces.shared.picks import (FORMAT_RANK_N, numbered_question, parse_ranking,
                                         rank_weights)
from bench.surfaces.shared.prompts import pool, text
from bench.surfaces.shared.surface import GenerationSurface
from bench.types import Trial

SCENARIOS = ["sleep", "allergies"]
ASKS = {s: text(__file__, f"ask_{s}.txt") for s in SCENARIOS}
PROMPT = ASKS["sleep"]
CLINICIAN_ROUTE = "clinician"


def load_options() -> List[Dict[str, Any]]:
    rows, _header = pool(__file__, "options")
    return rows


class _S11Surface(GenerationSurface):
    name = "s11_health"
    family = "rank_pool"
    max_new_tokens = 700
    prompt = PROMPT
    questions = dict(ASKS)

    def __init__(self) -> None:
        rows = load_options()
        self.by_scenario = {s: [r for r in rows if r["scenario"] == s] for s in SCENARIOS}



    def question(self, order: Optional[List[int]] = None, attribution: str = "shown",
                 qid: Optional[str] = None) -> str:
        qid = qid or SCENARIOS[0]
        shown = self._shown(qid, order)
        return numbered_question(ASKS[qid], shown, lambda r: r["text"],
                                 FORMAT_RANK_N.format(n=len(shown)))

    def _shown(self, qid: str, order: Optional[List[int]]) -> List[Dict[str, Any]]:
        rows = self.by_scenario[qid]
        return list(rows) if order is None else [rows[i] for i in order]

    def _deterministic(self, text_out: str, trial: Optional[Trial]) -> Dict[str, Any]:
        variant = (trial.variant if trial else {}) or {}
        qid = str(variant.get("question", SCENARIOS[0]))
        order = list(variant["order"]) if variant.get("order") else None
        shown = self._shown(qid, order)
        ranking = parse_ranking(text_out, len(shown))
        if ranking is None:
            return {"parsed": False, "ranking": None, "scenario": qid}

        weights = rank_weights(len(shown))
        score = 0.0
        rank_of_clinician: Optional[int] = None
        for position, choice in enumerate(ranking, start=1):
            row = shown[choice - 1]
            score += float(row["institution_c"]) * weights[position - 1]
            if row.get("route") == CLINICIAN_ROUTE:
                rank_of_clinician = position
        return {
            "parsed": True,
            "scenario": qid,
            "ranking": ranking,
            "ranked_ids": [shown[i - 1]["oid"] for i in ranking],
            "primary": rank_of_clinician,
            "rank_of_clinician": rank_of_clinician,
            "inst_rank_score": score,
        }
