"""s9_neighborhood -- shortlist three of ten places to live.

The pick-from-pool surface, ported from ``bench`` byte for byte so the two trees
can be compared: the ask, the pool, the numbering, the parser and the dependent
variable are the same strings and the same arithmetic.
``bench_v2/tests/test_all_tasks_parity.py`` asserts the built trials are equal.

Run the whole experiment, or stop at a stage::

    python -m bench_v2.tasks.s9_neighborhood.v1.pilot plan
    python -m bench_v2.tasks.s9_neighborhood.v1.pilot run --limit 4 --adaptor local_hf
    python -m bench_v2.tasks.s9_neighborhood.v1.pilot summary
    python -m bench_v2.tasks.s9_neighborhood.v1.pilot all       # run -> summary

``plan`` and ``summary`` are CPU-only; ``run`` needs whatever the adaptor needs.
Records go to ``runs/bench_v2/s9_neighborhood/v1``.
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
from bench_v2.helpers.readers import (
    FORMAT_PICK_N, detect_refusal, numbered_question, parse_picks, picked_mean,
    refusal_match, shown_mean, word_count,
)
from bench_v2.paths import items_dir, runs_dir
from bench_v2.types import Item, Outcome, Response, Trial

TASK = "s9_neighborhood"
VERSION = "v1"
SURFACE = TASK
TITLE = "shortlist three of ten places to live"
FAMILY = "pick_from_pool"
TASK_DIR = Path(__file__).resolve().parents[1]
PROMPT = render(TASK_DIR / "ask.j2")
ROWS, HEADER = load_pool(TASK_DIR / "neighborhoods.jsonl")
CONDITIONS: tuple[str, ...] = ("photos", "no_photos")
SCHEMES: tuple[str, ...] = ("chat", "agentic")
MAX_NEW_TOKENS = 700
N_PICKS = 3
ID_FIELD = "nid"
QUESTION_IDS = ["q0"]
OUT_DIR = Path(runs_dir()) / "bench_v2" / TASK / VERSION
DEFAULT_ITEMS = Path(items_dir()) / "explore_bucket_v1.jsonl"
DEFAULT_LIMIT = 4


def check_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    return condition


def is_item_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos"


# --- pool presentation -------------------------------------------------------
def _rows_for(qid: str | None) -> list[dict[str, Any]]:
    return ROWS


def _shown(order: list[int] | None, qid: str | None) -> list[dict[str, Any]]:
    rows = _rows_for(qid)
    return list(rows) if order is None else [rows[i] for i in order]


def _ask(qid: str | None) -> str:
    return PROMPT


def render_row(row: dict[str, Any]) -> str:
    return f"{row['name']} — {row['description']}"


def question_fn(qid: str, order: list[int] | None, attribution: str) -> str:
    shown = _shown(order, qid)
    return numbered_question(_ask(qid), shown, render_row,
                             FORMAT_PICK_N.format(n=N_PICKS))


# --- build -------------------------------------------------------------------
def variants() -> list[dict[str, Any]]:
    return [{"scheme": scheme, "question": qid}
            for scheme in SCHEMES for qid in QUESTION_IDS]


def build(item: Item, condition: str, variant: dict[str, Any] | None = None,
          seed: int = 42) -> Trial:
    condition = check_condition(condition)
    return build_base(
        surface=SURFACE, item=item, condition=condition, variant=variant, seed=seed,
        question_fn=question_fn, max_new_tokens=MAX_NEW_TOKENS, family=FAMILY,
        judge=None,
    )


# --- read --------------------------------------------------------------------
def _trial_qid(trial: Trial | None) -> str:
    variant = (trial.variant if trial else {}) or {}
    return str(variant.get("question", QUESTION_IDS[0]))


def _trial_shown(trial: Trial | None) -> list[dict[str, Any]]:
    variant = (trial.variant if trial else {}) or {}
    order = variant.get("order")
    return _shown(list(order) if order else None, _trial_qid(trial))


def _read_extra(trial: Trial | None) -> dict[str, Any]:
    return {}


def _dv(shown: list[dict[str, Any]], picks: list[int]) -> dict[str, Any]:
    return {
        "primary": picked_mean(shown, picks, "right_c") - shown_mean(shown, "right_c"),
        "right_pick_mean": picked_mean(shown, picks, "right_c"),
        "density_pick_mean": picked_mean(shown, picks, "density_c"),
        "composition_pick_mean": picked_mean(shown, picks, "composition_c"),
    }


def _read_deterministic(text: str, shown: list[dict[str, Any]],
                        trial: Trial | None) -> dict[str, Any]:
    picks = parse_picks(text, N_PICKS, len(shown))
    if picks is None:
        return {"parsed": False, "picks": None, **_read_extra(trial)}
    return {
        "parsed": True,
        "picks": picks,
        "picked_ids": [shown[i - 1][ID_FIELD] for i in picks],
        **_read_extra(trial),
        **_dv(shown, picks),
    }


def read(resp: Response, trial: Trial | None = None) -> Outcome:
    text = (resp.text or "").strip()
    extra = {
        "word_count": word_count(text),
        "refusal": detect_refusal(text),
        "refusal_match": refusal_match(text),
    }
    extra.update(_read_deterministic(text, _trial_shown(trial), trial))
    return Outcome(kind="generation", value=extra.get("primary"), extra=extra)


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
        for line in str(trial.meta.get("question", "")).splitlines()[:12]:
            print(f"    | {line}")
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
        print(f"{'condition':<16}{'n':>6}{'values':>9}{'mean':>10}"
              f"{'words':>9}{'refusals':>10}{'parsed':>9}")
        for condition in sorted(by_condition, key=str):
            group = by_condition[condition]
            values = [r["outcome"]["value"] for r in group
                      if r.get("outcome") and r["outcome"].get("value") is not None]
            words = [r["outcome"]["extra"].get("word_count") for r in group
                     if r.get("outcome") and r["outcome"].get("extra")]
            words = [w for w in words if w is not None]
            refusals = sum(1 for r in group
                           if r.get("outcome") and r["outcome"].get("extra", {}).get("refusal"))
            parsed = sum(1 for r in group
                         if r.get("outcome") and r["outcome"].get("extra", {}).get("parsed"))
            mean = f"{sum(values) / len(values):+.4f}" if values else "-"
            wmean = f"{sum(words) / len(words):.1f}" if words else "-"
            print(f"{str(condition):<16}{len(group):>6}{len(values):>9}{mean:>10}"
                  f"{wmean:>9}{refusals:>10}{parsed:>9}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
