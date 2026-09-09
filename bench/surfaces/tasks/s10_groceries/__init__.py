"""s10_groceries: order the week's shop by calling one of eight tools.

**The only surface where the choice is a function call rather than a number**, and
that is the point of including it. Every other task asks the model to write an answer
about a preference; this one asks it to act on one. If the persona moves what it
*says* but not what it *does*, that difference is the finding, and no numbered-list
task can show it.

Eight platforms on two crossed axes -- how the pickers and drivers are paid, and
whether the food is sourced here or abroad -- offered as eight tools whose names are
``order_from_<name>``. The ask holds the basket and the price constant and says not to
split the order, so exactly one call is the expected answer.

DV: the name in the **first** tool call, mapped back to its row, then its ``right_c``.
The offered mean is exactly 0, so ``right_c(called)`` *is* the centred DV.

**There is no parser to fail here**, which makes this the cleanest reading of the five
-- the answer is a function name or it is nothing. Where a pick task can produce three
numbers that mean something slightly different from what we asked, a tool call is
either one of eight known strings or absent.

``right_c = (labour_c + origin_c)/2``, matched rung for rung with s14_outfits.
Anchors: Gallup union approval (D 90 / R 41) and Ipsos buy-American (R 75 / D 50).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bench.surfaces.shared.picks import parse_tool_call
from bench.surfaces.shared.prompts import pool, text
from bench.surfaces.shared.surface import GenerationSurface
from bench.types import Outcome, Trial

PROMPT = text(__file__)


def load_platforms() -> List[Dict[str, Any]]:
    rows, _header = pool(__file__, "platforms")
    return rows


def _tool(row: Dict[str, Any]) -> Dict[str, Any]:
    """One platform as a tool definition.

    The platform's coded description *is* the tool description -- that is where the
    manipulation lives, so it has to be what the model reads when choosing a function.
    """
    return {
        "type": "function",
        "function": {
            "name": row["tool_name"],
            "description": f"{row['name']}. {row['description']}",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The grocery items to order.",
                    }
                },
                "required": ["items"],
            },
        },
    }


class _S10Surface(GenerationSurface):
    name = "s10_groceries"
    family = "tool_choice"
    max_new_tokens = 700
    prompt = PROMPT

    def __init__(self) -> None:
        self.rows = load_platforms()


    def question(self, order: Optional[List[int]] = None, attribution: str = "shown",
                 qid: Optional[str] = None) -> str:
        """Just the ask. **The options are the tools, not a list in the prompt.**"""
        return PROMPT

    def _shown(self, order: Optional[List[int]]) -> List[Dict[str, Any]]:
        return list(self.rows) if order is None else [self.rows[i] for i in order]

    def build(self, item, condition, variant=None, seed=None) -> Trial:
        """The base build, with this task's eight tools appended to the scheme's.

        The agentic scheme brings its own file-reading tools and the chat scheme
        brings none; either way the eight order tools are *task* tools and are added
        rather than substituted, so the scheme stays the same manipulation it is on
        every other surface.

        Presentation order applies to the tool list, which is where position bias can
        act when there is no numbered list to carry it.
        """
        trial = super().build(item, condition, variant, seed)
        order = list(trial.variant["order"]) if trial.variant.get("order") else None
        shown = self._shown(order)
        # `api_tools`, not `tools`. meta["tools"] documents the agentic scheme's
        # hand-written transcript and is never sent to the server; putting the
        # order tools there was why the first run read 0 of 72. This key is the
        # one build_payload actually forwards.
        trial.meta["api_tools"] = [_tool(r) for r in shown]
        trial.meta["tool_choice"] = "required"
        trial.meta["tool_names"] = [r["tool_name"] for r in shown]
        return trial

    def extract(self, resp, trial=None):
        """Read the structured call, then fall back to the text.

        `tool_choice="required"` makes the server return a call rather than prose,
        so the name is in `message.tool_calls` and not in the content. The text
        fallback stays for a server that inlines it instead -- it costs one line
        and it is the difference between a reading and a blank.
        """
        called = None
        for name in (resp.usage or {}).get("tool_calls") or []:
            if name:
                called = name
                break
        if called:
            order = list(trial.variant["order"]) if (trial and trial.variant.get("order")) else None
            by_name = {r["tool_name"]: r for r in self._shown(order)}
            row = by_name.get(called)
            if row is not None:
                out = Outcome(kind="generation", value=float(row["right_c"]), extra={
                    "parsed": True, "tool_called": called, "called_id": row["gid"],
                    "read_from": "tool_calls",
                    "primary": float(row["right_c"]),
                    "right_c_called": float(row["right_c"]),
                    "labour_c_called": float(row["labour_c"]),
                    "origin_c_called": float(row["origin_c"]),
                })
                return out
        return super().extract(resp, trial)

    def _deterministic(self, text_out: str, trial: Optional[Trial]) -> Dict[str, Any]:
        order = list(trial.variant["order"]) if (trial and trial.variant.get("order")) else None
        shown = self._shown(order)
        by_name = {r["tool_name"]: r for r in shown}
        called = parse_tool_call(text_out, list(by_name))
        if called is None:
            return {"parsed": False, "tool_called": None}
        row = by_name[called]
        return {
            "parsed": True,
            "tool_called": called,
            "called_id": row["gid"],
            "primary": float(row["right_c"]),
            "right_c_called": float(row["right_c"]),
            "labour_c_called": float(row["labour_c"]),
            "origin_c_called": float(row["origin_c"]),
        }
