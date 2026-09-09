"""s4_bonus -- how should the bonus pool be split.

The round-6 recipe, ported from ``bench`` byte for byte so the two trees can be
compared: the ask is the same string, the two conversation schemes are the same,
the reader is the same, and the judge spec hashes to the same ``judge_id``.

Run the whole experiment, or stop at a stage::

    python -m bench_v2.tasks.s4_bonus.v1.pilot plan
    python -m bench_v2.tasks.s4_bonus.v1.pilot run --limit 4 --adaptor local_hf
    python -m bench_v2.tasks.s4_bonus.v1.pilot judge
    python -m bench_v2.tasks.s4_bonus.v1.pilot summary
    python -m bench_v2.tasks.s4_bonus.v1.pilot all       # run -> judge -> summary

``plan`` and ``summary`` are CPU-only; ``judge`` needs an API key; ``run`` needs
whatever the adaptor needs. Records go to ``runs/bench_v2/s4_bonus/v1``.
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
from bench_v2.helpers.prompts import render
from bench_v2.helpers.readers import detect_refusal, refusal_match, word_count
from bench_v2.judge import aggregate_labels, attach_judge, judge_run
from bench_v2.paths import items_dir, runs_dir
from bench_v2.tasks.s4_bonus.judge_spec import JUDGE
from bench_v2.types import Item, Outcome, Response, Trial

TASK = "s4_bonus"
VERSION = "v1"
SURFACE = TASK
TITLE = "split the bonus pool"
TASK_DIR = Path(__file__).resolve().parents[1]
PROMPT = render(TASK_DIR / "prompt.j2")
CONDITIONS: tuple[str, ...] = ("photos", "no_photos")
SCHEMES: tuple[str, ...] = ("chat", "agentic")
MAX_NEW_TOKENS = 1000
OUT_DIR = Path(runs_dir()) / "bench_v2" / TASK / VERSION
DEFAULT_ITEMS = Path(items_dir()) / "explore_bucket_v1.jsonl"
DEFAULT_LIMIT = 4


def check_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    return condition


def is_item_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos"


def is_scheme_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos"


# --- build -------------------------------------------------------------------
def variants() -> list[dict[str, Any]]:
    return [{"scheme": scheme, "question": "q0"} for scheme in SCHEMES]


def question() -> str:
    return PROMPT


def build(item: Item, condition: str, variant: dict[str, Any] | None = None,
          seed: int = 42) -> Trial:
    condition = check_condition(condition)
    return build_base(
        surface=SURFACE, item=item, condition=condition, variant=variant, seed=seed,
        question_fn=lambda qid, order, attribution: question(),
        max_new_tokens=MAX_NEW_TOKENS, judge=JUDGE.id,
    )


def read(resp: Response, trial: Trial | None = None) -> Outcome:
    text = (resp.text or "").strip()
    extra = {
        "word_count": word_count(text),
        "refusal": detect_refusal(text),
        "refusal_match": refusal_match(text),
    }
    return Outcome(kind="generation", value=extra.get("primary"), extra=extra)


# --- main --------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{TITLE} pilot")
    parser.add_argument("phase", nargs="?", default="plan",
                        choices=["plan", "run", "judge", "summary", "all"])
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

    if args.phase in {"judge", "all"}:
        # --- stage 4: judge the answers (needs an API key) -------------------
        judge_run(OUT_DIR, JUDGE, limit=args.limit)
        attach_judge(OUT_DIR, JUDGE)   # fold the labels into trials.jsonl

    if args.phase in {"summary", "all"}:
        # --- stage 5: summarize the dependent variable ----------------------
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
        print(f"{'condition':<16}{'n':>6}{'words':>9}{'refusals':>10}")
        for condition in sorted(by_condition, key=str):
            group = by_condition[condition]
            words = [r["outcome"]["extra"].get("word_count") for r in group
                     if r.get("outcome") and r["outcome"].get("extra")]
            words = [w for w in words if w is not None]
            refusals = sum(1 for r in group
                           if r.get("outcome") and r["outcome"].get("extra", {}).get("refusal"))
            wmean = f"{sum(words) / len(words):.1f}" if words else "-"
            print(f"{str(condition):<16}{len(group):>6}{wmean:>9}{refusals:>10}")

        judged = OUT_DIR / "judged.jsonl"
        if judged.exists():
            print("\njudge (mean, mapped to [-1, 1] / [-2, 2])")
            for field, stats in aggregate_labels(OUT_DIR, JUDGE).items():
                if field.startswith("_"):
                    continue
                mean = f"{stats['mean']:+.4f}" if stats["mean"] is not None else "-"
                print(f"  {field:<26}{mean:>10}  (n={stats['n']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
