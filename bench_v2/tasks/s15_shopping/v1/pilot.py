"""s15_shopping -- a search-and-recommend environment over real shops.

Ported from ``bench/surfaces/tasks/s15_shopping`` so the build is byte-identical:
the ask is the same string, the same five search tools and one ``ask_user`` tool
are offered, and the dependent variable is the ``right_c`` of the venue
recommended first.

Run the whole experiment, or stop at a stage::

    python -m bench_v2.tasks.s15_shopping.v1.pilot plan
    python -m bench_v2.tasks.s15_shopping.v1.pilot run --limit 4 --adaptor local_hf
    python -m bench_v2.tasks.s15_shopping.v1.pilot summary
    python -m bench_v2.tasks.s15_shopping.v1.pilot all

``plan`` and ``summary`` are CPU-only; ``run`` needs whatever the adaptor needs.
Records go to ``runs/bench_v2/s15_shopping/v1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_v2 import registry
from bench_v2.helpers import run as run_helper
from bench_v2.helpers.agentloop import run_agent
from bench_v2.helpers.dataset import read_items
from bench_v2.helpers.generation import build_base
from bench_v2.helpers.prompts import load_pool, render
from bench_v2.paths import items_dir, runs_dir
from bench_v2.types import Item, Outcome, Response, Trial

TASK = "s15_shopping"
VERSION = "v1"
SURFACE = TASK
TITLE = "search real shops and recommend one"
TASK_DIR = Path(__file__).resolve().parents[1]
PROMPT = render(TASK_DIR / "prompt.j2")
ASK_USER_TOOL = "ask_user"
# Zero. Every venue returns the same total.
PRICE_OFFSETS = [0.0, 0.0, 0.0, 0.0, 0.0]
CONDITIONS: tuple[str, ...] = ("photos", "no_photos")
SCHEMES: tuple[str, ...] = ("chat", "agentic")
FAMILY = "search_and_recommend"
MAX_NEW_TOKENS = 900
OUT_DIR = Path(runs_dir()) / "bench_v2" / TASK / VERSION
DEFAULT_ITEMS = Path(items_dir()) / "explore_bucket_v1.jsonl"
DEFAULT_LIMIT = 4
REMIND = "Call the ask_user tool with your recommendation."

ROWS, HEADER = load_pool(TASK_DIR / "venues.jsonl")


def check_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    return condition


def is_item_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos"


def is_scheme_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos"


# --- the environment ---------------------------------------------------------
def shown_rows(order: Optional[list[int]] = None) -> list[dict[str, Any]]:
    return list(ROWS) if not order else [ROWS[i] for i in order]


def search_tool(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": row["tool_name"],
            "description": f"Search {row['name']} for grocery items and prices. {row['blurb']}",
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


def ask_user_tool(names: list[str]) -> dict[str, Any]:
    """The readout. The DV is ``recommended``, so it is a closed enum -- a free
    string would let the model answer with a shop that was never offered."""
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


def tools(order: Optional[list[int]] = None) -> list[dict[str, Any]]:
    shown = shown_rows(order)
    return [search_tool(r) for r in shown] + [ask_user_tool([r["name"] for r in shown])]


class _ShoppingEnv:
    """The tools s15 offers, and what each one returns."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def priced(self, row: dict[str, Any], offset: float) -> dict[str, Any]:
        """One venue's listing, with the trial's price offset applied."""
        items = [{"item": it["item"], "price_usd": round(it["base_price_usd"] * (1 + offset), 2)}
                 for it in row["items"]]
        return {"store": row["name"], "note": row["blurb"], "items": items,
                "basket_total_usd": round(sum(i["price_usd"] for i in items), 2)}

    def offsets_for(self, rotation: int) -> dict[str, float]:
        """Which venue gets which price offset on this trial."""
        n = len(self.rows)
        return {self.rows[i]["tool_name"]: PRICE_OFFSETS[(i + rotation) % n] for i in range(n)}

    def call(self, name: str, arguments: dict[str, Any], rotation: int) -> str:
        """Run one tool call and return what the model should see back."""
        offsets = self.offsets_for(rotation)
        for row in self.rows:
            if row["tool_name"] == name:
                return json.dumps(self.priced(row, offsets[name]), ensure_ascii=False)
        return json.dumps({"error": f"no such tool: {name}"})


# --- reading -----------------------------------------------------------------
def read_recommendation(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """The DV, from the agent's own tool calls.

    The first ``ask_user`` is the answer; a run that never calls it has no
    recommendation and is **not** guessed at from the prose.
    """
    by_name = {r["name"]: r for r in ROWS}
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
            "searched_all": len(set(searched)) == len(ROWS),
            "is_local": row["right_c"] is None,
        }
        if row["right_c"] is not None:
            out["primary"] = float(row["right_c"])
            out["right_c_recommended"] = float(row["right_c"])
        return out
    return {"parsed": False, "why": "never called ask_user",
            "n_searches": len(searched), "searched": searched}


# --- build -------------------------------------------------------------------
def variants() -> list[dict[str, Any]]:
    return [{"scheme": scheme, "question": "q0"} for scheme in SCHEMES]


def question() -> str:
    return PROMPT


def meta_extra(variant: dict[str, Any], order: Optional[list[int]],
               qid: str) -> dict[str, Any]:
    return {
        "api_tools": tools(order),
        "tool_choice": "auto",
        "agent_loop": True,
        "price_rotation": int(variant.get("price_rotation", 0)),
        "venue_order": [r["name"] for r in shown_rows(order)],
    }


def build(item: Item, condition: str, variant: dict[str, Any] | None = None,
          seed: int = 42) -> Trial:
    condition = check_condition(condition)
    return build_base(
        surface=SURFACE, item=item, condition=condition, variant=variant, seed=seed,
        question_fn=lambda qid, order, attribution: question(),
        max_new_tokens=MAX_NEW_TOKENS, family=FAMILY, judge=None,
        meta_extra=meta_extra,
    )


def read(resp: Response, trial: Trial | None = None) -> Outcome:
    calls = (resp.usage or {}).get("agent_calls") or []
    extra = read_recommendation(calls)
    # The old AgentSurface.extract hardcodes value=None; the DV rides in extra.
    return Outcome(kind="generation", value=None, extra=extra)


def run_agent_cells(adaptor: Any, trial: Trial, env: _ShoppingEnv) -> Response:
    calls, _transcript, error = run_agent(
        adaptor, trial, env, terminal=ASK_USER_TOOL, remind=REMIND)
    return Response(text="", usage={"agent_calls": calls}, error=error)


# --- main --------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{TITLE} pilot")
    parser.add_argument("phase", nargs="?", default="plan",
                        choices=["plan", "run", "summary", "all"])
    parser.add_argument("--items", default=DEFAULT_ITEMS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--scheme", action="append", default=None, help="repeatable; default all")
    parser.add_argument("--adaptor", default="local_hf")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cells", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- stage 1: load the items ---------------------------------------------
    items, synthetic = read_items(args.items, args.limit)

    # --- stage 2: enumerate the cells (condition x scheme x item) ------------
    chosen_variants = [v for v in variants()
                       if not args.scheme or str(v.get("scheme")) in args.scheme]
    cells = run_helper.cell_plan(CONDITIONS, chosen_variants, items, is_item_invariant)

    if args.phase == "plan":
        # --- stage 2b: print what would run, and one example question (CPU) ---
        print(f"{TITLE}  [{SURFACE}/{VERSION}]")
        print(f"  {len(items)} items x {len(CONDITIONS)} conditions x "
              f"{len(chosen_variants)} variants -> {len(cells)} trials"
              + ("  (synthetic, no images)" if synthetic else ""))
        condition, variant, item = cells[0]
        trial = build(item, condition, dict(variant), seed=args.seed)
        print(f"  example: condition={condition} variant={trial.variant_key}")
        print(f"  tools: {len(trial.meta.get('api_tools', []))}")
        return 0

    if args.phase in {"run", "all"}:
        # --- stage 3: drive the agent loop over the environment -------------
        registry.load_adaptors()
        adaptor_cls = registry.get_adaptor(args.adaptor)
        kwargs = {"seed": args.seed}
        if args.model:
            kwargs["model"] = args.model
        adaptor = adaptor_cls(**kwargs)
        env = _ShoppingEnv(ROWS)
        run_helper.run_cells(
            surface=SURFACE, cells=cells, build=build, read=read, adaptor=adaptor,
            out_dir=OUT_DIR, seed=args.seed, limit_cells=args.limit_cells,
            note=f"{TASK}/{VERSION}",
            run_agent=lambda adaptor, trial: run_agent_cells(adaptor, trial, env),
        )

    if args.phase in {"summary", "all"}:
        # --- stage 4: summarize the dependent variable ----------------------
        trials = OUT_DIR / "trials.jsonl"
        if not trials.exists():
            print(f"no records at {trials}; run `pilot run` first")
            return 1
        rows = [json.loads(line) for line in trials.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_condition[row.get("condition")].append(row)
        print(f"{TITLE}  [{SURFACE}/{VERSION}]  {len(rows)} records")
        print(f"{'condition':<16}{'n':>6}{'values':>9}{'mean':>10}{'parsed':>9}{'local':>7}")
        for condition in sorted(by_condition, key=str):
            group = by_condition[condition]
            values = [r["outcome"]["value"] for r in group
                      if r.get("outcome") and r["outcome"].get("value") is not None]
            parsed = sum(1 for r in group
                         if r.get("outcome") and r["outcome"].get("extra", {}).get("parsed"))
            local = sum(1 for r in group
                        if r.get("outcome") and r["outcome"].get("extra", {}).get("is_local"))
            mean = f"{sum(values) / len(values):+.4f}" if values else "-"
            print(f"{str(condition):<16}{len(group):>6}{len(values):>9}{mean:>10}"
                  f"{parsed:>9}{local:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
