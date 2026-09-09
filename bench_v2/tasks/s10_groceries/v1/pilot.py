"""s10_groceries -- order the week's shop by calling one of eight tools.

Ported from ``bench/surfaces/tasks/s10_groceries`` so the build is byte-identical:
the ask is the same string, the eight platform tools are the same definitions, and
the dependent variable is the ``right_c`` of the platform whose tool was called.

Run the whole experiment, or stop at a stage::

    python -m bench_v2.tasks.s10_groceries.v1.pilot plan
    python -m bench_v2.tasks.s10_groceries.v1.pilot run --limit 4 --adaptor local_hf
    python -m bench_v2.tasks.s10_groceries.v1.pilot summary
    python -m bench_v2.tasks.s10_groceries.v1.pilot all

``plan`` and ``summary`` are CPU-only; ``run`` needs whatever the adaptor needs.
Records go to ``runs/bench_v2/s10_groceries/v1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_v2 import registry
from bench_v2.helpers import run as run_helper
from bench_v2.helpers.dataset import read_items
from bench_v2.helpers.generation import build_base
from bench_v2.helpers.prompts import load_pool, render
from bench_v2.helpers.readers import parse_tool_call
from bench_v2.paths import items_dir, runs_dir
from bench_v2.types import Item, Outcome, Response, Trial

TASK = "s10_groceries"
VERSION = "v1"
SURFACE = TASK
TITLE = "order the shop by calling one of eight tools"
TASK_DIR = Path(__file__).resolve().parents[1]
PROMPT = render(TASK_DIR / "prompt.j2")
CONDITIONS: tuple[str, ...] = ("photos", "no_photos")
SCHEMES: tuple[str, ...] = ("chat", "agentic")
FAMILY = "tool_choice"
MAX_NEW_TOKENS = 700
OUT_DIR = Path(runs_dir()) / "bench_v2" / TASK / VERSION
DEFAULT_ITEMS = Path(items_dir()) / "explore_bucket_v1.jsonl"
DEFAULT_LIMIT = 4

ROWS, HEADER = load_pool(TASK_DIR / "platforms.jsonl")


def check_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    return condition


def is_item_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos"


def is_scheme_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos"


# --- the pool ----------------------------------------------------------------
def shown_rows(order: list[int] | None = None) -> list[dict[str, Any]]:
    return list(ROWS) if not order else [ROWS[i] for i in order]


def tool_for(row: dict[str, Any]) -> dict[str, Any]:
    """One platform as a tool definition.

    The platform's coded description *is* the tool description -- that is where
    the manipulation lives, so it has to be what the model reads when choosing a
    function.
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


def tool_extra(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "called_id": row["gid"],
        "primary": float(row["right_c"]),
        "right_c_called": float(row["right_c"]),
        "labour_c_called": float(row["labour_c"]),
        "origin_c_called": float(row["origin_c"]),
    }


# --- build -------------------------------------------------------------------
def variants() -> list[dict[str, Any]]:
    return [{"scheme": scheme, "question": "q0"} for scheme in SCHEMES]


def question() -> str:
    return PROMPT


def meta_extra(variant: dict[str, Any], order: list[int] | None,
               qid: str) -> dict[str, Any]:
    shown = shown_rows(order)
    return {
        "api_tools": [tool_for(r) for r in shown],
        "tool_choice": "required",
        "tool_names": [r["tool_name"] for r in shown],
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
    """The DV: the first tool call, structured first, then the text fallback."""
    order = (trial.variant if trial else {}).get("order") if trial else None
    shown = shown_rows(order)
    by_name = {r["tool_name"]: r for r in shown}
    for name in (resp.usage or {}).get("tool_calls") or []:
        row = by_name.get(name)
        if row is not None:
            extra = {"parsed": True, "tool_called": name, "read_from": "tool_calls",
                     **tool_extra(row)}
            return Outcome(kind="generation", value=extra["primary"], extra=extra)
    called = parse_tool_call((resp.text or "").strip(), list(by_name))
    if called is None:
        return Outcome(kind="generation", value=None,
                       extra={"parsed": False, "tool_called": None})
    row = by_name[called]
    extra = {"parsed": True, "tool_called": called, **tool_extra(row)}
    return Outcome(kind="generation", value=extra["primary"], extra=extra)


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
        # --- stage 3: run the generations -----------------------------------
        registry.load_adaptors()
        adaptor_cls = registry.get_adaptor(args.adaptor)
        kwargs = {"seed": args.seed}
        if args.model:
            kwargs["model"] = args.model
        adaptor = adaptor_cls(**kwargs)
        run_helper.run_cells(
            surface=SURFACE, cells=cells, build=build, read=read, adaptor=adaptor,
            out_dir=OUT_DIR, seed=args.seed, limit_cells=args.limit_cells,
            note=f"{TASK}/{VERSION}",
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
        print(f"{'condition':<16}{'n':>6}{'values':>9}{'mean':>10}{'parsed':>9}")
        for condition in sorted(by_condition, key=str):
            group = by_condition[condition]
            values = [r["outcome"]["value"] for r in group
                      if r.get("outcome") and r["outcome"].get("value") is not None]
            parsed = sum(1 for r in group
                         if r.get("outcome") and r["outcome"].get("extra", {}).get("parsed"))
            mean = f"{sum(values) / len(values):+.4f}" if values else "-"
            print(f"{str(condition):<16}{len(group):>6}{len(values):>9}{mean:>10}{parsed:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
