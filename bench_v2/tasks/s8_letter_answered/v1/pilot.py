"""s8_letter_answered -- the letter to a representative, with the model's own
question answered.

Ported from ``bench/surfaces/tasks/s8_letter_answered`` byte for byte: the same
twelve concerns in the same order, the same s5 opening ask, the same scripted
assistant turn and prefill, and a judge that hashes to the same ``s2_proposal``
rubric. ``bench_v2/tests/test_all_tasks_parity.py`` asserts the build matches.

Run the whole experiment, or stop at a stage::

    python -m bench_v2.tasks.s8_letter_answered.v1.pilot plan
    python -m bench_v2.tasks.s8_letter_answered.v1.pilot run --limit 4 --adaptor local_hf
    python -m bench_v2.tasks.s8_letter_answered.v1.pilot judge
    python -m bench_v2.tasks.s8_letter_answered.v1.pilot summary
    python -m bench_v2.tasks.s8_letter_answered.v1.pilot all   # run -> judge -> summary

``plan`` and ``summary`` are CPU-only; ``judge`` needs an API key; ``run`` needs
whatever the adaptor needs. Records go to ``runs/bench_v2/s8_letter_answered/v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_v2 import registry
from bench_v2.helpers import run as run_helper
from bench_v2.helpers.dataset import read_items
from bench_v2.helpers.generation import build_base
from bench_v2.helpers.prompts import load_json, load_pool, render
from bench_v2.helpers.readers import detect_refusal, refusal_match, word_count
from bench_v2.judge import aggregate_labels, attach_judge, judge_run
from bench_v2.paths import items_dir, runs_dir
from bench_v2.tasks.s8_letter_answered.judge_spec import JUDGE
from bench_v2.types import Item, Outcome, Response, Trial

TASK = "s8_letter_answered"
VERSION = "v1"
SURFACE = TASK
TITLE = "write a letter to my representative about a supplied concern"
TASK_DIR = Path(__file__).resolve().parents[1]
CONDITIONS: tuple[str, ...] = ("photos", "no_photos")
SCHEMES: tuple[str, ...] = ("chat", "agentic")
MAX_NEW_TOKENS = 800
OUT_DIR = Path(runs_dir()) / "bench_v2" / TASK / VERSION
DEFAULT_ITEMS = Path(items_dir()) / "explore_bucket_v1.jsonl"
DEFAULT_LIMIT = 4

ROW_META = ("domain", "topic", "lean", "concern")

# The opening ask is s5's prompt, copied here so this task is self-contained. The
# scripted assistant turn is the model's own words (a real s5 reply), trimmed to
# the one question the dataset answers. The prefill is the model's own opening,
# json because its trailing blank line is the mechanism.
OPENING_ASK = render(TASK_DIR / "prompt.j2")
ASSISTANT_ASKS = render(TASK_DIR / "assistant_asks.j2")
PREFILL = load_json(TASK_DIR / "prefill.json")["prefill"]

_ROWS, _HEADER = load_pool(TASK_DIR / "concerns_v1.jsonl")
BY_ID: dict[str, dict[str, Any]] = {row["cid"]: row for row in _ROWS}
QUESTION_IDS: tuple[str, ...] = tuple(row["cid"] for row in _ROWS)
DATASET_VERSION = str(_HEADER.get("version", "unknown"))


def dataset_fingerprint(rows: Sequence[dict[str, Any]], id_field: str,
                        text_field: str) -> str:
    """sha256 over (id, text) in file order -- only what reaches the model."""
    payload = [[row[id_field], row[text_field]] for row in rows]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=False,
                      separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


DATASET_HASH = dataset_fingerprint(_ROWS, "cid", "concern")


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
    return [{"scheme": scheme, "question": qid}
            for scheme in SCHEMES for qid in QUESTION_IDS]


def question(qid: str) -> str:
    """The first user turn is always s5's opening ask; the concern comes later."""
    return OPENING_ASK


def meta_extra(variant: dict[str, Any], order: list[int] | None,
               qid: str) -> dict[str, Any]:
    row = BY_ID.get(qid, {})
    return {
        "dataset": {
            "version": DATASET_VERSION,
            "hash": DATASET_HASH,
            "cid": qid,
            **{field: row.get(field) for field in ROW_META},
        }
    }


def build(item: Item, condition: str, variant: dict[str, Any] | None = None,
          seed: int = 42) -> Trial:
    condition = check_condition(condition)
    cid = str((variant or {}).get("question", ""))
    if cid not in BY_ID:
        raise ValueError(f"{SURFACE}: unknown concern {cid!r}; known: {sorted(BY_ID)}")

    trial = build_base(
        surface=SURFACE, item=item, condition=condition, variant=variant, seed=seed,
        question_fn=lambda qid, order, attribution: question(qid),
        max_new_tokens=MAX_NEW_TOKENS, prefill=PREFILL, judge=JUDGE.id,
        meta_extra=meta_extra,
    )
    row = BY_ID[cid]

    messages = list(trial.conversation.messages)
    if condition == "no_photos":
        # The no-photos condition strips the *persona* framing, not the task's own
        # shape, so the opening request is restored here.
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": OPENING_ASK}]}]
    messages.append({"role": "assistant",
                     "content": [{"type": "text", "text": ASSISTANT_ASKS}]})
    messages.append({"role": "user",
                     "content": [{"type": "text", "text": row["concern"]}]})
    trial = replace(trial, conversation=replace(trial.conversation, messages=messages))

    trial.meta["question"] = row["concern"]      # what the model is answering
    trial.meta["opening_ask"] = OPENING_ASK
    trial.meta["assistant_asks"] = ASSISTANT_ASKS
    trial.meta["prefix_n_messages"] = len(messages) - 1
    return trial


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

    # --- stage 2: enumerate the cells (condition x scheme x question) --------
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
