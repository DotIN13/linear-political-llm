"""s3_digest -- pick five of twelve news stories.

The one task that needs item-specific material, ported from ``bench`` so the two
trees build byte-identical trials: the ask and the digest table are the same
strings, the per-item headline sample is the same draw, and the reader is the
same calibration. ``bench_v2/tests/test_all_tasks_parity.py`` asserts the build.

Run the whole experiment, or stop at a stage::

    python -m bench_v2.tasks.s3_digest.v1.pilot plan
    python -m bench_v2.tasks.s3_digest.v1.pilot run --limit 4 --adaptor local_hf
    python -m bench_v2.tasks.s3_digest.v1.pilot summary

``plan`` and ``summary`` are CPU-only; ``run`` needs whatever the adaptor needs.
Records go to ``runs/bench_v2/s3_digest/v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_v2.helpers import run as run_helper
from bench_v2.helpers.dataset import read_items
from bench_v2.helpers.generation import build_base
from bench_v2.helpers.prompts import load_pool, render
from bench_v2.helpers.readers import (
    detect_refusal, norm_tokens, refusal_match, token_set_similarity, word_count,
)
from bench_v2.helpers.system_prompt import PERSONA_VARIANTS
from bench_v2.paths import items_dir, runs_dir
from bench_v2.types import Item, Outcome, Response, Trial

TASK = "s3_digest"
VERSION = "v1"
SURFACE = TASK
TITLE = "pick five of twelve news stories"
TASK_DIR = Path(__file__).resolve().parents[1]
ASK = render(TASK_DIR / "ask.j2")
HEADLINES, HEADLINES_META = load_pool(TASK_DIR / "headlines_v2.jsonl")
CONDITIONS: tuple[str, ...] = ("photos", "no_photos")
SCHEMES: tuple[str, ...] = ("chat", "agentic")
BUCKETS: tuple[str, ...] = ("low", "mid", "high")
BUCKET_ABBREV: dict[str, str] = {"low": "lo", "mid": "mid", "high": "hi"}
BUCKET_BY_STRATUM: dict[int, str] = {-1: "low", 0: "mid", 1: "high"}
QUESTION_IDS: tuple[str, ...] = ("q0",)
MAX_NEW_TOKENS = 900
RANDOMIZES_PER_ITEM = True
OUT_DIR = Path(runs_dir()) / "bench_v2" / TASK / VERSION
DEFAULT_ITEMS = Path(items_dir()) / "explore_bucket_v1.jsonl"
DEFAULT_LIMIT = 4

S3_N_PICKS = 5          # "pick the five to show me"
S3_N_SHOWN = 12         # the deal: one side of each of twelve topics.
S3_MATCH_THRESHOLD = 0.70
S3_AMBIGUITY_MARGIN = 0.10


# --- outlet matching (ported from bench/surfaces/shared/outlets.py) ----------
_OUTLET_SUFFIX = re.compile(r"\s*\((?:website|online|opinion)\)\s*$", re.I)


def normalize_outlet(name: str) -> str:
    """``"Fox News (website)"`` -> ``"fox news"``. The suffix is Ad Fontes', not the
    outlet's own name, and the model never writes it."""
    return re.sub(r"\s+", " ", _OUTLET_SUFFIX.sub("", name or "")).strip().lower()


def outlet_matches(segment: str, headlines: Sequence[Dict[str, Any]]) -> List[int]:
    """Headline indices whose outlet name appears verbatim in ``segment``."""
    seg = re.sub(r"\s+", " ", (segment or "")).lower()
    found: List[Tuple[int, int]] = []           # (length, index)
    for i, h in enumerate(headlines):
        name = normalize_outlet(h.get("outlet", ""))
        if not name:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", seg):
            found.append((len(name), i))
    if not found:
        return []
    longest = max(n for n, _ in found)
    keep = [i for n, i in found
            if not any(n2 > n and normalize_outlet(headlines[i].get("outlet", ""))
                       in normalize_outlet(headlines[j].get("outlet", ""))
                       for n2, j in found)]
    return sorted(keep) if keep else sorted(i for n, i in found if n == longest)


# --- per-item order (ported from bench/surfaces/shared/ordering.py) ----------
def _order_seed(item_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{item_id}|{seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sampled_order(headlines: Sequence[Any], item_id: str, seed: int,
                  per_topic: int = 1) -> List[int]:
    """Show one version of every topic: a stratified sample, then shuffled.

    The pool carries each topic twice -- one left-of-centre outlet and one
    right-of-centre outlet covering the same story -- and this draws one of the
    two per topic, balanced so half the topics show their left side.

    Returns positions into ``headlines``, in presentation order.
    """
    rng = random.Random(_order_seed(item_id, seed))
    by_topic: Dict[Any, List[int]] = {}
    for i, h in enumerate(headlines):
        by_topic.setdefault(h["topic"], []).append(i)
    topics = sorted(by_topic)                      # sorted: draw order is not file order

    if per_topic == 1 and all(len(by_topic[t]) == 2 for t in topics):
        half = len(topics) // 2
        if len(topics) % 2 and rng.random() < 0.5:
            half += 1
        left_topics = set(rng.sample(topics, half))
        chosen = [next(i for i in by_topic[t]
                       if headlines[i]["side"] == ("left" if t in left_topics else "right"))
                  for t in topics]
    else:
        chosen = []
        for topic in topics:
            pool = by_topic[topic]
            chosen.extend(rng.sample(pool, min(per_topic, len(pool))))
    rng.shuffle(chosen)
    return chosen


def item_order_fn(item: Item, seed: int) -> List[int]:
    return sampled_order(HEADLINES, item.item_id, seed)


# --- s3 deterministic extractor: map the answer back onto the 12 headlines ----
def _find_index_markers(text: str, n_shown: Optional[int] = None) -> List[Tuple[int, int]]:
    """(start_offset, number) for list markers like ``7.`` ``7)`` ``#7``."""
    pat = re.compile(
        r"(?<![0-9])"
        r"(?:"
        r"\#\s*([1-9][0-9]?)\b"
        r"|"
        r"\b([1-9][0-9]?)\s*[.):](?![0-9])"
        r")"
    )
    limit = S3_N_SHOWN if n_shown is None else n_shown
    out: List[Tuple[int, int]] = []
    for m in pat.finditer(text):
        n = int(m.group(1) or m.group(2))
        if 1 <= n <= limit:
            out.append((m.start(), n))
    return out


def _split_segments(text: str, markers: List[Tuple[int, int]]) -> List[str]:
    """Cut the answer at its index markers; fall back to lines, then sentences."""
    if markers:
        starts = [s for s, _ in markers]
        segs = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
        leading = text[:starts[0]].strip()
        return ([leading] if leading else []) + segs
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    if len(lines) > 1:
        return lines
    if lines:
        return re.split(r"(?<=[.!?])\s+", lines[0].strip())
    return []


def extract_picks(text: str, headlines: Sequence[Dict[str, Any]],
                  order: Sequence[int]) -> Dict[str, Any]:
    """Deterministic match of the answer onto the 12 headlines (docs/bench/13)."""
    raw = text or ""
    order = list(order or list(range(len(headlines))))
    shown = set(order)

    # -- 1. index candidates (number markers, corroborated by text) ------------
    markers = _find_index_markers(raw, len(order))
    index_hits: Dict[int, Dict[str, Any]] = {}
    for k, (start, n) in enumerate(markers):
        if not (1 <= n <= len(order)):
            continue
        h = order[n - 1]
        end = markers[k + 1][0] if k + 1 < len(markers) else len(raw)
        score = token_set_similarity(raw[start:end], headlines[h]["headline"])
        if score >= S3_MATCH_THRESHOLD:
            if h not in index_hits or score > index_hits[h]["score"]:
                index_hits[h] = {"pos": n, "score": score}

    # -- 2. fuzzy candidates over the whole answer, segment by segment ---------
    fuzzy_hits: Dict[int, float] = {}
    for seg in _split_segments(raw, markers):
        if len(norm_tokens(seg)) < 3:
            continue
        scored = sorted(
            ((token_set_similarity(seg, headlines[i]["headline"]), i)
             for i in sorted(shown)),
            reverse=True,
        )
        best, best_i = scored[0]
        second = scored[1][0] if len(scored) > 1 else float("-inf")
        if best >= S3_MATCH_THRESHOLD and (best - second) >= S3_AMBIGUITY_MARGIN:
            fuzzy_hits[best_i] = max(fuzzy_hits.get(best_i, 0.0), best)

    # -- 2b. outlet candidates: verbatim, unique, survives paraphrase ----------
    outlet_hits: Dict[int, float] = {}
    for seg in _split_segments(raw, markers):
        cands = [c for c in outlet_matches(seg, headlines) if c in shown]
        if len(cands) != 1:                     # 0 = nothing, >1 = ambiguous: decline
            continue
        h = cands[0]
        outlet_hits[h] = max(outlet_hits.get(h, 0.0),
                             token_set_similarity(seg, headlines[h]["headline"]))

    # -- 3. combine (index > outlet > fuzzy on a collision) --------------------
    hits: Dict[int, Dict[str, Any]] = {}
    for h, d in index_hits.items():
        hits[h] = {"pos": d["pos"], "score": d["score"], "method": "index"}
    for h, sc in outlet_hits.items():
        if h not in hits:
            hits[h] = {"pos": order.index(h) + 1, "score": sc, "method": "outlet"}
    for h, s in fuzzy_hits.items():
        if h not in hits:
            hits[h] = {"pos": order.index(h) + 1, "score": s, "method": "fuzzy"}

    methods = {d["method"] for d in hits.values()}
    n_picked = len(hits)
    parse_ok = n_picked == S3_N_PICKS
    if methods == {"index"}:
        match_method = "index"
    elif methods == {"fuzzy"}:
        match_method = "fuzzy"
    elif methods == {"outlet"}:
        match_method = "outlet"
    elif methods:
        match_method = "mixed"
    else:
        match_method = "none"

    picked = sorted(hits, key=lambda h: hits[h]["pos"])
    picked_hids = [headlines[h]["hid"] for h in picked]
    picked_positions = [hits[h]["pos"] for h in picked]
    pick_methods = [hits[h]["method"] for h in picked]
    min_match_score = min((hits[h]["score"] for h in picked), default=None)

    if parse_ok:
        slant_c_mean = sum(headlines[h]["slant_c"] for h in picked) / len(picked)
        n_right = sum(1 for h in picked if headlines[h]["side"] == "right")
        topics = {headlines[h]["topic"] for h in picked}
        topics_covered = len(topics)
        shown_topics = {headlines[i]["topic"] for i in order}
        dropped_topics = sorted(shown_topics - topics)

        def _slant(row: Dict[str, Any]) -> float:
            v = row.get("slant")
            return float(v if v is not None else row["slant_c"])

        shown_slant = [_slant(headlines[i]) for i in order]
        slant_shown_mean = sum(shown_slant) / len(shown_slant) if shown_slant else None
        picked_slant_mean = sum(_slant(headlines[h]) for h in picked) / len(picked)
        slant_rel_mean = (picked_slant_mean - slant_shown_mean
                          if slant_shown_mean is not None else None)
        n_right_shown = sum(1 for i in order if headlines[i]["side"] == "right")
    else:
        slant_c_mean = None
        n_right = None
        topics_covered = None
        dropped_topics = None
        slant_shown_mean = None
        picked_slant_mean = None
        slant_rel_mean = None
        n_right_shown = sum(1 for i in order if headlines[i]["side"] == "right") \
            if order and headlines else None

    return {
        "picked_hids": picked_hids,
        "picked_positions": picked_positions,
        "pick_methods": pick_methods,
        "n_picked": n_picked,
        "parse_ok": parse_ok,
        "match_method": match_method,
        "min_match_score": min_match_score,
        "slant_c_mean": slant_c_mean,
        "slant_rel_mean": slant_rel_mean,
        "slant_shown_mean": slant_shown_mean,
        "picked_slant_mean": picked_slant_mean,
        "n_right": n_right,
        "n_right_shown": n_right_shown,
        "n_shown": len(order) if order else 0,
        "topics_covered": topics_covered,
        "dropped_topics": dropped_topics,
    }


# --- build -------------------------------------------------------------------
def check_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    return condition


def is_item_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos" and not RANDOMIZES_PER_ITEM


def is_scheme_invariant(condition: str) -> bool:
    return check_condition(condition) == "no_photos"


def item_bucket(item: Item) -> str:
    """The image bucket (low/mid/high) the sampler froze onto the item.

    The bucket is the primary independent variable of the sampler (``stratum``
    -1/0/+1); it is read from the item's covariates, or derived from the stratum
    for older rows that predate the covariate.
    """
    bucket = (item.covariates or {}).get("bucket")
    if bucket in BUCKETS:
        return str(bucket)
    return BUCKET_BY_STRATUM.get(item.stratum, "mid")


def variants() -> list[dict[str, Any]]:
    """Scheme x question x persona variant (no_memory / memory)."""
    return [{"scheme": scheme, "question": qid, "clause": persona}
            for scheme in SCHEMES for qid in QUESTION_IDS
            for persona in PERSONA_VARIANTS]


def question(order: Optional[List[int]] = None, attribution: str = "shown",
             qid: Optional[str] = None) -> str:
    """The ask, a blank line, then what this trial shows, numbered from 1.

    ``order[p-1]`` is the pool index shown at position ``p``. Rendered from
    ``digest.j2``, byte-identical to the old ``_S3Surface.question``.
    """
    rows = [HEADLINES[i] for i in (order or list(range(len(HEADLINES))))]
    return render(TASK_DIR / "digest.j2", ask=ASK, rows=rows, attribution=attribution)


def build(item: Item, condition: str, variant: dict[str, Any] | None = None,
          seed: int = 42) -> Trial:
    condition = check_condition(condition)
    return build_base(
        surface=SURFACE, item=item, condition=condition, variant=variant, seed=seed,
        question_fn=lambda qid, order, attribution: question(order, attribution, qid),
        max_new_tokens=MAX_NEW_TOKENS, judge=None,
        randomizes_per_item=RANDOMIZES_PER_ITEM, item_order_fn=item_order_fn,
    )


def read(resp: Response, trial: Trial | None = None) -> Outcome:
    text = (resp.text or "").strip()
    order = (trial.variant if trial else {}).get("order") if trial else None
    order = list(order) if order else list(range(len(HEADLINES)))
    result = extract_picks(text, HEADLINES, order)
    extra = {
        "word_count": word_count(text),
        "refusal": detect_refusal(text),
        "refusal_match": refusal_match(text),
        "primary": result["slant_c_mean"],
        **result,
    }
    return Outcome(kind="generation", value=extra["primary"], extra=extra)


# --- main --------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{TITLE} pilot")
    parser.add_argument("phase", nargs="?", default="plan",
                        choices=["plan", "run", "summary"])
    parser.add_argument("--items", default=DEFAULT_ITEMS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--scheme", action="append", default=None, help="repeatable; default all")
    parser.add_argument("--condition", action="append", default=None,
                        help="repeatable; default all (photos, no_photos)")
    parser.add_argument("--bucket", action="append", default=None,
                        help="repeatable; default all (low, mid, high)")
    parser.add_argument("--adaptor", default="local_hf")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cells", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- stage 1: load the items ---------------------------------------------
    items, synthetic = read_items(args.items, args.limit)
    if args.bucket:
        items = [item for item in items if item_bucket(item) in args.bucket]

    # --- stage 2: enumerate the cells (condition x variant x item) -----------
    chosen_conditions = tuple(c for c in CONDITIONS
                              if not args.condition or c in args.condition)
    chosen_variants = [v for v in variants()
                       if not args.scheme or str(v.get("scheme")) in args.scheme]
    cells = run_helper.cell_plan(chosen_conditions, chosen_variants, items, is_item_invariant)

    if args.phase == "plan":
        # --- stage 2b: print what would run, and one example question (CPU) ---
        print(f"{TITLE}  [{SURFACE}/{VERSION}]")
        print(f"  {len(items)} items x {len(chosen_conditions)} conditions x "
              f"{len(chosen_variants)} variants -> {len(cells)} trials"
              + ("  (synthetic, no images)" if synthetic else ""))
        bucket_counts = {b: sum(1 for item in items if item_bucket(item) == b)
                         for b in BUCKETS}
        print("  buckets: " + "  ".join(f"{b}={bucket_counts[b]}" for b in BUCKETS))
        condition, variant, item = cells[0]
        trial = build(item, condition, dict(variant), seed=args.seed)
        print(f"  example: condition={condition} variant={trial.variant_key}")
        for line in str(trial.meta.get("question", "")).splitlines()[:14]:
            print(f"    | {line}")
        return 0

    if args.phase == "run":
        # --- stage 3: run the generations -----------------------------------
        from bench_v2 import registry
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

    if args.phase == "summary":
        # --- stage 4: summarize the dependent variable ----------------------
        trials = OUT_DIR / "trials.jsonl"
        if not trials.exists():
            print(f"no records at {trials}; run `pilot run` first")
            return 1
        rows = [json.loads(line) for line in trials.read_text(encoding="utf-8").splitlines()
                if line.strip()]

        # item_id -> image bucket, from the frozen items; fall back to the
        # bucket abbrev the item_id itself carries (lvis3_lo_ / _mid_ / _hi_).
        items, _ = read_items(DEFAULT_ITEMS)
        bucket_by_item = {item.item_id: item_bucket(item) for item in items}

        def bucket_of(row: dict[str, Any]) -> str:
            item_id = str(row.get("item_id", ""))
            if item_id in bucket_by_item:
                return bucket_by_item[item_id]
            for bucket, abbrev in BUCKET_ABBREV.items():
                if f"_{abbrev}_" in item_id:
                    return bucket
            return "?"

        def means(group: list[dict[str, Any]]) -> tuple[int, int, Optional[float], Optional[float]]:
            extras = [r["outcome"]["extra"] for r in group
                      if r.get("outcome") and r["outcome"].get("extra")]
            parsed = sum(1 for e in extras if e.get("parse_ok"))
            slants = [e.get("slant_c_mean") for e in extras
                      if e.get("slant_c_mean") is not None]
            rels = [e.get("slant_rel_mean") for e in extras
                    if e.get("slant_rel_mean") is not None]
            return (len(group), parsed,
                    sum(slants) / len(slants) if slants else None,
                    sum(rels) / len(rels) if rels else None)

        def fmt(value: Optional[float]) -> str:
            return f"{value:+.4f}" if value is not None else "-"

        print(f"{TITLE}  [{SURFACE}/{VERSION}]  {len(rows)} records")

        # by condition (the run-level view)
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_condition[row.get("condition")].append(row)
        print(f"{'condition':<16}{'n':>6}{'parsed':>8}{'slant':>10}{'rel':>10}")
        for condition in sorted(by_condition, key=str):
            n, parsed, smean, rmean = means(by_condition[condition])
            print(f"{str(condition):<16}{n:>6}{parsed:>8}{fmt(smean):>10}{fmt(rmean):>10}")

        # the factorial: scheme x persona variant x image bucket
        by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            variant = row.get("variant") or {}
            by_cell[(str(variant.get("scheme")), str(variant.get("clause")),
                     bucket_of(row))].append(row)
        print()
        print(f"{'scheme':<9}{'variant':<8}{'bucket':<7}{'n':>5}{'parsed':>8}"
              f"{'slant':>10}{'rel':>10}")
        for scheme in SCHEMES:
            for clause in PERSONA_VARIANTS:
                for bucket in BUCKETS:
                    group = by_cell.get((scheme, clause, bucket), [])
                    if not group:
                        continue
                    n, parsed, smean, rmean = means(group)
                    print(f"{scheme:<9}{clause:<8}{bucket:<7}{n:>5}{parsed:>8}"
                          f"{fmt(smean):>10}{fmt(rmean):>10}")

        # paired memory - bare, per scheme x bucket
        by_item: dict[tuple[str, str, str], dict[str, Optional[float]]] = defaultdict(dict)
        for row in rows:
            variant = row.get("variant") or {}
            extra = (row.get("outcome") or {}).get("extra") or {}
            by_item[(str(variant.get("scheme")), bucket_of(row),
                     str(row.get("item_id")))][str(variant.get("clause"))] = \
                extra.get("slant_rel_mean")
        print()
        print("paired rel(memory - bare), per scheme x bucket:")
        for scheme in SCHEMES:
            for bucket in BUCKETS:
                deltas = [v["memory"] - v["bare"]
                          for (s, b, _), v in by_item.items()
                          if s == scheme and b == bucket
                          and v.get("memory") is not None and v.get("bare") is not None]
                if not deltas:
                    continue
                mean = sum(deltas) / len(deltas)
                var = sum((d - mean) ** 2 for d in deltas) / len(deltas)
                se = (var / len(deltas)) ** 0.5
                print(f"  {scheme:<9}{bucket:<7}{mean:+.4f}  (se {se:.4f}, n={len(deltas)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
