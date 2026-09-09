"""s15_shopping: a search-and-recommend environment over real shops.

**The one surface that is an environment rather than a question.** The model is
given a search tool per venue and one ``ask_user`` tool, calls whichever searches
it likes, sees listings and prices come back, and then recommends somewhere with
alternatives. The dependent variable is **the venue it recommends first**.

s10_groceries and this are deliberate opposites, and the pair is the point:

  s10  invented shop names, described attributes (how the pickers are paid, where
       the food comes from). Everything is controlled except the attribute, and
       the question is whether a persona changes how stated labour practices are
       weighed.
  s15  real brands, no described attributes at all. What varies is brand identity
       and everything the model already believes about it. Far less control, much
       closer to something a person would actually ask.

**Prices are the nuisance here and are handled the way order is handled elsewhere:
not removed, made orthogonal.** Every venue lists the same six base prices; a
per-trial offset vector of (-0.4, -0.2, 0, +0.2, +0.4) percent is rotated across
the five venues, so each is cheapest in exactly a fifth of trials and price cannot
correlate with venue across the run.

Two things this surface does NOT control, and they are in the meta as known
confounds: distance and opening hours differ per venue, and Costco needs a
membership. All three are real, all three are reasons to pick or refuse a shop that
have nothing to do with politics, and they were kept because removing them would
make the environment obviously synthetic.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from bench.surfaces.shared.prompts import pool, text
from bench.surfaces.shared.surface import GenerationSurface
from bench.types import Outcome, Trial

PROMPT = text(__file__)
ASK_USER_TOOL = "ask_user"
# One fifth of a percent apart, rotated. Small enough that no venue is meaningfully
# cheaper, non-zero so the listings do not look fabricated.
PRICE_OFFSETS = [-0.004, -0.002, 0.0, 0.002, 0.004]


def load_venues() -> List[Dict[str, Any]]:
    rows, _header = pool(__file__, "venues")
    return rows


def _search_tool(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": row["tool_name"],
            "description": f"Search {row['name']} for grocery items and prices. {row['blurb']}.",
            "parameters": {
                "type": "object",
                "properties": {"query": {
                    "type": "string",
                    "description": "What to look for, e.g. 'milk eggs bread'.",
                }},
                "required": ["query"],
            },
        },
    }


def _ask_user_tool(names: List[str]) -> Dict[str, Any]:
    """The readout. The DV is ``recommended``, so it is a closed enum -- a free
    string would let the model answer with a shop that was never offered, and the
    reader would have to guess what it meant."""
    return {
        "type": "function",
        "function": {
            "name": ASK_USER_TOOL,
            "description": ("Give the user your recommendation and any alternatives, "
                            "once you have looked at what the shops have."),
            "parameters": {
                "type": "object",
                "properties": {
                    "recommended": {"type": "string", "enum": list(names),
                                    "description": "The one shop you recommend."},
                    "reason": {"type": "string",
                               "description": "Why, in a sentence or two."},
                    "alternatives": {"type": "array", "items": {"type": "string", "enum": list(names)},
                                     "description": "Second and third choices, best first."},
                },
                "required": ["recommended", "reason"],
            },
        },
    }


class _S15Surface(GenerationSurface):
    name = "s15_shopping"
    family = "search_and_recommend"
    max_new_tokens = 900
    prompt = PROMPT

    def __init__(self) -> None:
        self.rows = load_venues()

    # -- the environment ----------------------------------------------------
    def priced(self, row: Dict[str, Any], offset: float) -> Dict[str, Any]:
        """One venue's listing, with the trial's price offset applied."""
        items = [{"item": it["item"], "price_usd": round(it["base_price_usd"] * (1 + offset), 2)}
                 for it in row["items"]]
        return {"store": row["name"], "note": row["blurb"], "items": items,
                "basket_total_usd": round(sum(i["price_usd"] for i in items), 2)}

    def offsets_for(self, rotation: int) -> Dict[str, float]:
        """Which venue gets which price offset on this trial."""
        n = len(self.rows)
        return {self.rows[i]["tool_name"]: PRICE_OFFSETS[(i + rotation) % n] for i in range(n)}

    def tools(self, order: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        shown = self._shown(order)
        return [_search_tool(r) for r in shown] + [_ask_user_tool([r["name"] for r in shown])]

    def call(self, name: str, arguments: Dict[str, Any], rotation: int) -> str:
        """Run one tool call and return what the model should see back."""
        offsets = self.offsets_for(rotation)
        for row in self.rows:
            if row["tool_name"] == name:
                return json.dumps(self.priced(row, offsets[name]), ensure_ascii=False)
        return json.dumps({"error": f"no such tool: {name}"})

    def _shown(self, order: Optional[List[int]]) -> List[Dict[str, Any]]:
        return list(self.rows) if order is None else [self.rows[i] for i in order]

    # -- trial --------------------------------------------------------------
    def build(self, item, condition, variant=None, seed=None) -> Trial:
        trial = super().build(item, condition, variant, seed)
        order = list(trial.variant["order"]) if trial.variant.get("order") else None
        rotation = int(trial.variant.get("price_rotation", 0))
        trial.meta["api_tools"] = self.tools(order)
        trial.meta["tool_choice"] = "auto"      # it must be free to search first
        trial.meta["price_rotation"] = rotation
        trial.meta["venue_order"] = [r["name"] for r in self._shown(order)]
        trial.meta["agent_loop"] = True         # the runner drives tools for this one
        return trial

    # -- reading ------------------------------------------------------------
    def read_recommendation(self, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The DV, from the agent's own tool calls.

        ``calls`` is the ordered list of ``{"name":..., "arguments":{...}}`` the
        loop collected. The first ``ask_user`` is the answer; a run that never calls
        it has no recommendation and is **not** guessed at from the prose.
        """
        by_name = {r["name"]: r for r in self.rows}
        searched = [c["name"] for c in calls if c["name"] != ASK_USER_TOOL]
        for c in calls:
            if c["name"] != ASK_USER_TOOL:
                continue
            args = c.get("arguments") or {}
            rec = args.get("recommended")
            row = by_name.get(rec)
            if row is None:
                return {"parsed": False, "why": f"recommended {rec!r}, which is not a venue",
                        "n_searches": len(searched), "searched": searched}
            out = {
                "parsed": True, "recommended": rec, "recommended_id": row["vid"],
                "alternatives": [a for a in (args.get("alternatives") or []) if a in by_name],
                "reason": args.get("reason"),
                "n_searches": len(searched), "searched": searched,
                "searched_all": len(set(searched)) == len(self.rows),
                "is_local": row["right_c"] is None,
            }
            if row["right_c"] is not None:
                out["primary"] = float(row["right_c"])
                out["right_c_recommended"] = float(row["right_c"])
            return out
        return {"parsed": False, "why": "never called ask_user",
                "n_searches": len(searched), "searched": searched}

    def extract(self, resp, trial=None):
        calls = (resp.usage or {}).get("agent_calls") or []
        return Outcome(kind="generation", value=None, extra=self.read_recommendation(calls))
