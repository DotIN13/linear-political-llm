"""s9_neighborhood v2 -- recommend five of sixteen places to live, ranked.

v1 was the round-6 recipe ported from ``bench`` byte for byte: ten options built
from five density rungs crossed with two levels of a *community composition*
clause ("almost everyone here grew up nearby" against "others came in the last
decade"). The metadata cites Mummolo & Nall for that second axis, but the paper
measured stated preference for racial and ethnic diversity, so the citation and
the clause describe different things. v1 also runs ``chat``/``agentic`` with no
persona factor at all, and ``variants()`` never sets ``order``, so the pool is
shown in file order in every trial however the metadata describes it.

v2 changes four things:

* **Four evidenced attributes, fully crossed.** Access, faith, composition and
  water. Every option is a corner of the 2^4 cube, so the pool mean is exactly 0
  on each attribute. Each attribute is a measured partisan gap, cited in
  ``neighborhoods_v2.meta.json``; composition is now written as the shops on the
  high street, which is what Pew's 56-point diversity gap actually measures.
* **A per-attribute reading.** The composite ``right_rank_w`` still answers "how
  right-coded was the shortlist"; the four per-attribute means say *which*
  attribute carried it, which v1's single density number could not.
* **Five ranked recommendations** instead of three unranked picks, so the
  primary reading is rank-weighted.
* **The two gates in.** The full scheme set (``chat``/``agentic``/
  ``agentic_live``) crossed with the persona instruction (``bare``/``memory``),
  and an ``item_order_fn`` so position is a near-complete Latin square.

Run the whole experiment, or stop at a stage::

    python -m bench_v2.tasks.s9_neighborhood.v2.pilot plan
    python -m bench_v2.tasks.s9_neighborhood.v2.pilot run --limit 4 --adaptor local_hf
    python -m bench_v2.tasks.s9_neighborhood.v2.pilot summary
    python -m bench_v2.tasks.s9_neighborhood.v2.pilot all

``plan`` and ``summary`` are CPU-only; ``run`` needs whatever the adaptor needs.
The dependent variable is deterministic, so there is no judge. Records go to
``runs/bench_v2/s9_neighborhood/v2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_v2 import registry
from bench_v2.helpers import run as run_helper
from bench_v2.helpers.dataset import read_items, take_per_bucket
from bench_v2.helpers.generation import build_base
from bench_v2.helpers.prompts import load_pool
from bench_v2.helpers.readers import (
    FORMAT_RANK_N, detect_refusal, parse_picks, picked_mean,
    refusal_match, word_count,
)
from bench_v2.helpers.system_prompt import PERSONA_VARIANTS
from bench_v2.paths import items_dir, runs_dir
from bench_v2.types import Item, Outcome, Response, Trial

TASK = "s9_neighborhood"
VERSION = "v2"
SURFACE = TASK
TITLE = "recommend five of sixteen places to live"
FAMILY = "pick_from_pool"
HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent
ROWS, HEADER = load_pool(HERE / "neighborhoods_v2.jsonl")

CONDITIONS: tuple[str, ...] = ("photos", "no_photos")
SCHEMES: tuple[str, ...] = ("chat", "agentic", "agentic_live")
BUCKETS: tuple[str, ...] = ("low", "mid", "high")
BUCKET_ABBREV: dict[str, str] = {"low": "lo", "mid": "mid", "high": "hi"}
BUCKET_BY_STRATUM: dict[int, str] = {-1: "low", 0: "mid", 1: "high"}
# In the order they are read out, so the summary table and the file agree.
AXES: tuple[str, ...] = ("access", "faith", "composition", "water")
N_PICKS = 5
PICK_WEIGHTS: tuple[int, ...] = tuple(range(N_PICKS, 0, -1))   # 5,4,3,2,1
WEIGHT_SUM = sum(PICK_WEIGHTS)
ID_FIELD = "nid"
QUESTION_IDS: tuple[str, ...] = ("q0",)
MAX_NEW_TOKENS = 1600
RANDOMIZES_PER_ITEM = True
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
# The wording is not here. Each scheme's request is a component,
# ``<scheme>/request.j2``, which contains the framing and then ``{{ question }}``
# where this body goes. A task with no such file gets the default: the body, and
# nothing else. That is what makes an override a file drop rather than a code edit.


def _rows_for(qid: str | None) -> list[dict[str, Any]]:
    return ROWS


def _shown(order: list[int] | None, qid: str | None) -> list[dict[str, Any]]:
    rows = _rows_for(qid)
    return list(rows) if order is None else [rows[i] for i in order]


def render_row(row: dict[str, Any]) -> str:
    return f"{row['name']} — {row['description']}"


def question_fn(qid: str, order: list[int] | None, attribution: str) -> str:
    """The body: the sixteen options numbered, then the format line.

    The framing is not here, because it depends on the scheme and this signature does
    not receive one. Each scheme's ``request.j2`` holds the framing and a
    ``{{ question }}`` where this goes.
    """
    shown = _shown(order, qid)
    lines = [f"{i + 1}. {render_row(row)}" for i, row in enumerate(shown)]
    return "\n".join(lines) + FORMAT_RANK_N.format(n=N_PICKS)


def item_order_fn(item: Item, seed: int) -> list[int]:
    """A cyclic rotation of the fixed pool order, crossed with a reversal arm.

    v1's metadata describes exactly this and never builds it: ``variants()``
    returns only ``scheme`` and ``question``, so ``build_base`` sees no pinned
    order, takes ``randomizes_per_item`` Falsely and passes ``order=None``, and
    every trial shows the pool in file order. Sixteen options in a thousand-word
    prompt is a lot of room for a position effect, so v2 supplies the function.

    The rotation index is a hash of the item id, which makes rotation independent
    of the item's bucket -- otherwise position would be confounded with the very
    thing the buckets are for. The reversal arm trades first place for last
    between items.
    """
    n = len(ROWS)
    digest = hashlib.sha256(f"{item.item_id}|{seed}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:4], "big") % n
    order = [(start + i) % n for i in range(n)]
    if digest[4] % 2:
        order.reverse()
    return order


# --- build -------------------------------------------------------------------
def variants() -> list[dict[str, Any]]:
    return [{"scheme": scheme, "clause": variant, "question": qid}
            for scheme in SCHEMES
            for variant in PERSONA_VARIANTS
            for qid in QUESTION_IDS]


def build(item: Item, condition: str, variant: dict[str, Any] | None = None,
          seed: int = 42) -> Trial:
    condition = check_condition(condition)

    def meta_extra(v: dict[str, Any], _order, _qid) -> dict[str, Any]:
        scheme = str((v or {}).get("scheme", "chat"))
        return {"framing": scheme}

    return build_base(
        surface=SURFACE, item=item, condition=condition, variant=variant, seed=seed,
        question_fn=question_fn, max_new_tokens=MAX_NEW_TOKENS, family=FAMILY,
        judge=None, randomizes_per_item=RANDOMIZES_PER_ITEM,
        item_order_fn=item_order_fn, meta_extra=meta_extra,
        # One argument for the whole prompt layer: this task's chat/, agentic/ and
        # agentic_live/ component files are picked up from here, and a <scheme>.py
        # beside this pilot would replace that scheme's shape outright. Neither
        # exists, so every arm is the global default with this task's wording.
        task_dir=HERE,
    )


# --- read --------------------------------------------------------------------
def _trial_qid(trial: Trial | None) -> str:
    variant = (trial.variant if trial else {}) or {}
    return str(variant.get("question", QUESTION_IDS[0]))


def _trial_shown(trial: Trial | None) -> list[dict[str, Any]]:
    variant = (trial.variant if trial else {}) or {}
    order = variant.get("order")
    return _shown(list(order) if order else None, _trial_qid(trial))


def _weighted(shown: list[dict[str, Any]], picks: Sequence[int], field: str) -> float:
    """Rank-weighted mean: the first recommendation counts five times the last."""
    return sum(float(shown[p - 1][field]) * w
               for p, w in zip(picks, PICK_WEIGHTS)) / WEIGHT_SUM


def _dv(shown: list[dict[str, Any]], picks: Sequence[int]) -> dict[str, Any]:
    """The composite, plus one reading per attribute.

    Every pool mean is exactly 0 by construction (all sixteen are always shown),
    so no shown-mean subtraction is needed and 0 is "no preference" for every
    column.
    """
    out: dict[str, Any] = {
        "primary": _weighted(shown, picks, "right_c"),
        "right_rank_w": _weighted(shown, picks, "right_c"),
        "right_pick_mean": picked_mean(shown, picks, "right_c"),
    }
    for axis in AXES:
        out[f"{axis}_pick_mean"] = picked_mean(shown, picks, f"{axis}_c")
        out[f"{axis}_rank_w"] = _weighted(shown, picks, f"{axis}_c")
    return out


def _read_deterministic(text: str, shown: list[dict[str, Any]],
                        trial: Trial | None) -> dict[str, Any]:
    picks = parse_picks(text, N_PICKS, len(shown))
    if picks is None:
        return {"parsed": False, "picks": None}
    return {
        "parsed": True,
        "picks": picks,
        "picked_ids": [shown[i - 1][ID_FIELD] for i in picks],
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


# --- summary -----------------------------------------------------------------
def item_bucket(item: Item) -> str:
    """The image bucket the sampler froze onto the item, as low / mid / high."""
    bucket = (item.covariates or {}).get("bucket")
    if bucket in BUCKETS:
        return str(bucket)
    return BUCKET_BY_STRATUM.get(item.stratum, "mid")


def bucket_of(row: dict[str, Any], bucket_by_item: dict[str, str]) -> str:
    """The bucket of a finished trial, from the item id or the items file.

    Item ids carry a ``_lo_`` / ``_mid_`` / ``_hi_`` segment, the same convention
    s1 and s5 read, so a summary works from ``trials.jsonl`` alone.
    """
    item_id = str(row.get("item_id") or "")
    if item_id in bucket_by_item:
        return bucket_by_item[item_id]
    for bucket, abbrev in BUCKET_ABBREV.items():
        if f"_{abbrev}_" in item_id:
            return bucket
    return "mid"


def paired(rows: list[dict[str, Any]], bucket_by_item: dict[str, str],
           field: str) -> dict[tuple[str, str], tuple[float, float, int]]:
    """Per-item paired ``memory - bare``, keyed by (scheme, bucket).

    Only items with both arms contribute, which is what makes it a within-item
    comparison rather than a difference of two means over different items.
    """
    by_arm: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row.get("condition") != "photos":
            continue
        outcome = row.get("outcome") or {}
        extra = outcome.get("extra") or {}
        value = extra.get(field)
        if value is None:
            continue
        variant = row.get("variant") or {}
        by_arm[(str(variant.get("scheme")), bucket_of(row, bucket_by_item),
                str(row.get("item_id")), str(variant.get("clause")))] = float(value)
    out: dict[tuple[str, str], tuple[float, float, int]] = {}
    for scheme in SCHEMES:
        for bucket in BUCKETS:
            deltas = [value - by_arm[(scheme, bucket, item_id, "bare")]
                      for (s, b, item_id, clause), value in by_arm.items()
                      if s == scheme and b == bucket and clause == "memory"
                      and (scheme, bucket, item_id, "bare") in by_arm]
            if not deltas:
                continue
            mean = sum(deltas) / len(deltas)
            var = (sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
                   if len(deltas) > 1 else 0.0)
            out[(scheme, bucket)] = (mean, (var / len(deltas)) ** 0.5, len(deltas))
    return out


def _load_rows(out_dir: Path) -> list[dict[str, Any]]:
    trials = out_dir / "trials.jsonl"
    if not trials.exists():
        return []
    return [json.loads(line) for line in trials.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def print_summary(rows: list[dict[str, Any]], items_path: str | Path) -> None:
    print(f"{TITLE}  [{SURFACE}/{VERSION}]  {len(rows)} records")
    if not rows:
        return
    items, _synthetic = read_items(items_path, 0)
    bucket_by_item = {item.item_id: item_bucket(item) for item in items}

    # --- means by scheme x variant x bucket ---------------------------------
    group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("condition") != "photos":
            continue
        variant = row.get("variant") or {}
        key = (str(variant.get("scheme")), str(variant.get("clause")),
               bucket_of(row, bucket_by_item))
        group[key].append(row)

    cols = ("right_rank_w",) + AXES
    header = f"{'scheme':<14}{'clause':<9}{'bucket':<8}{'n':>5}"
    header += "".join(f"{c[:11]:>13}" for c in cols)
    print(header)
    for scheme in SCHEMES:
        for clause in PERSONA_VARIANTS:
            for bucket in BUCKETS:
                rows_here = group.get((scheme, clause, bucket), [])
                if not rows_here:
                    continue
                line = f"{scheme:<14}{clause:<9}{bucket:<8}{len(rows_here):>5}"
                for col in cols:
                    vals = [r["outcome"]["extra"].get(col) for r in rows_here
                            if r.get("outcome") and r["outcome"].get("extra")
                            and r["outcome"]["extra"].get(col) is not None]
                    line += f"{sum(vals) / len(vals):>+13.4f}" if vals else f"{'-':>13}"
                print(line)

    # --- the hypothesis: paired memory - bare, per scheme x bucket ----------
    print()
    print("paired memory - bare, within item (positive = the shortlist moved right)")
    print(f"{'field':<18}{'scheme':<14}{'bucket':<8}{'delta':>10}{'se':>9}{'n':>6}")
    for field in (("right_rank_w",) + tuple(f"{a}_pick_mean" for a in AXES)):
        for scheme in SCHEMES:
            stats = paired(rows, bucket_by_item, field)
            for bucket in BUCKETS:
                if (scheme, bucket) not in stats:
                    continue
                mean, se, n = stats[(scheme, bucket)]
                print(f"{field:<18}{scheme:<14}{bucket:<8}{mean:>+10.4f}{se:>9.4f}{n:>6}")


# --- main --------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{TITLE} pilot")
    parser.add_argument("phase", nargs="?", default="plan",
                        choices=["plan", "run", "summary", "all"])
    parser.add_argument("--items", default=str(DEFAULT_ITEMS))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="items, taken from the head of the file; ignored when --per-bucket is set")
    parser.add_argument("--per-bucket", type=int, default=0,
                        help="items per image-lean bucket; an items file is grouped by bucket, so "
                             "--limit alone would draw every item from one of them")
    parser.add_argument("--scheme", action="append", default=None, help="repeatable; default all")
    parser.add_argument("--clause", action="append", default=None, help="bare / memory; repeatable")
    parser.add_argument("--bucket", action="append", default=None, help="low / mid / high; repeatable")
    parser.add_argument("--adaptor", default="local_hf")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cells", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out) if args.out else OUT_DIR

    # --- stage 1: load the items ---------------------------------------------
    items, synthetic = read_items(args.items, 0 if args.per_bucket else args.limit)
    if args.per_bucket:
        items = take_per_bucket(items, args.per_bucket, item_bucket, BUCKETS)
    if args.bucket:
        items = [item for item in items if item_bucket(item) in set(args.bucket)]

    # --- stage 2: enumerate the cells (condition x scheme x clause x item) ---
    chosen_variants = [v for v in variants()
                       if (not args.scheme or str(v.get("scheme")) in args.scheme)
                       and (not args.clause or str(v.get("clause")) in args.clause)]
    cells = run_helper.cell_plan(CONDITIONS, chosen_variants, items, is_item_invariant)

    if args.phase == "plan":
        # --- stage 2b: print what would run, and one example question (CPU) ---
        print(f"{TITLE}  [{SURFACE}/{VERSION}]")
        print(f"  {len(items)} items x {len(CONDITIONS)} conditions x "
              f"{len(chosen_variants)} variants -> {len(cells)} trials"
              + ("  (synthetic, no images)" if synthetic else ""))
        counts = {b: sum(1 for item in items if item_bucket(item) == b) for b in BUCKETS}
        print("  buckets: " + "  ".join(f"{b}={counts[b]}" for b in BUCKETS))
        condition, variant, item = cells[0]
        trial = build(item, condition, dict(variant), seed=args.seed)
        question = str(trial.meta.get("question", ""))
        lines = question.splitlines()
        print(f"  example: condition={condition} variant={trial.variant_key}")
        print(f"  pool words: {len(question.split())}   options: {len(ROWS)}   picks: {N_PICKS}")
        for line in lines[:4]:
            print(f"    | {line[:150]}")
        print(f"    | ... {len(lines) - 5} more ...")
        print(f"    | {lines[-1]}")
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
            out_dir=out_dir, seed=args.seed, limit_cells=args.limit_cells,
            note=f"{TASK}/{VERSION}", workers=args.workers,
        )

    if args.phase in {"summary", "all"}:
        # --- stage 4: summarize the dependent variable ----------------------
        rows = _load_rows(out_dir)
        if not rows:
            print(f"no records at {out_dir / 'trials.jsonl'}; run `pilot run` first")
            return 1
        print_summary(rows, args.items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
