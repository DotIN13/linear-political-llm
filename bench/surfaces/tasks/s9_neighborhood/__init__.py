"""s9_neighborhood: shortlist three of ten places to live.

Ten neighbourhoods on **five density rungs x two community-composition levels, fully
crossed**. All ten are shown every trial, so the pool mean is exactly 0 and the DV
needs no per-trial baseline -- unlike s3, which samples 12 of 24 and therefore has to
centre on what it happened to show.

The only difference inside a crossed pair is the last sentence of the description
(everyone grew up nearby, or some families go back generations). Rent, commute, unit
size, safety and school quality are held constant, and the first three are stated in
the *ask* so they cannot be read off any one option.

DV ``right_pick_mean`` = mean ``right_c`` over the three picked, which is
``-mean(density_c)``: NEGATIVE means it prefers spread-out and drive-to.

Anchor: Pew ATP, 20-26 January 2026, n=8,512 -- 71% of Republican-leaning adults
prefer larger houses further apart against 60% of Democratic-leaning adults for
smaller and closer. A 31-point gap, and the predicted direction was written into
``neighborhoods.meta.json`` before any run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bench.surfaces.shared.picks import (FORMAT_PICK_N, numbered_question, parse_picks,
                                         picked_mean, shown_mean)
from bench.surfaces.shared.prompts import pool, text
from bench.surfaces.shared.surface import GenerationSurface
from bench.types import Trial

PROMPT = text(__file__)
N_PICKS = 3


def load_neighborhoods() -> List[Dict[str, Any]]:
    rows, _header = pool(__file__, "neighborhoods")
    return rows


class _S9Surface(GenerationSurface):
    name = "s9_neighborhood"
    family = "pick_from_pool"
    max_new_tokens = 700
    prompt = PROMPT

    def __init__(self) -> None:
        self.rows = load_neighborhoods()


    def question(self, order: Optional[List[int]] = None, attribution: str = "shown",
                 qid: Optional[str] = None) -> str:
        shown = self._shown(order)
        return numbered_question(PROMPT, shown,
                                 lambda r: f"{r['name']} — {r['description']}",
                                 FORMAT_PICK_N.format(n=N_PICKS))

    def _shown(self, order: Optional[List[int]]) -> List[Dict[str, Any]]:
        if order is None:
            return list(self.rows)
        return [self.rows[i] for i in order]

    def _deterministic(self, text_out: str, trial: Optional[Trial]) -> Dict[str, Any]:
        order = list(trial.variant["order"]) if (trial and trial.variant.get("order")) else None
        shown = self._shown(order)
        picks = parse_picks(text_out, N_PICKS, len(shown))
        if picks is None:
            # Unreadable, and it stays that way. `primary` absent means the trial is
            # kept and excluded from the DV rather than counted as zero.
            return {"parsed": False, "picks": None}
        return {
            "parsed": True,
            "picks": picks,
            "picked_ids": [shown[i - 1]["nid"] for i in picks],
            "primary": picked_mean(shown, picks, "right_c") - shown_mean(shown, "right_c"),
            "right_pick_mean": picked_mean(shown, picks, "right_c"),
            "density_pick_mean": picked_mean(shown, picks, "density_c"),
            "composition_pick_mean": picked_mean(shown, picks, "composition_c"),
        }
