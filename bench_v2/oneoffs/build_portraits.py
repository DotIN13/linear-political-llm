"""Freeze a pool of EasyPortrait portraits, bucketed by the political-lean probe.

EasyPortrait is a set of close-up portraits; the *subject* is the user themselves,
which is why v5 attaches one as context next to the persona photos. Each portrait
has a per-image probe score (``image_mean``) on the same ``combined_ideology``
headwise-linear probe the LVIS item buckets use, so "low / mid / high" mean the
same direction in both.

The scores are all bunched above zero (min ~ -0.4, median ~ +0.2), unlike the LVIS
images which straddle the fixed +/-0.5 cut, so the buckets here are **rank
tertiles** of the portrait corpus, not the fixed thresholds. The pool records both
the raw score and the rank bucket, so an analysis can ignore the bucket and use
the score continuously.

Run once, commit ``portraits_v1.jsonl`` + ``portraits_v1.meta.json``::

    python -m bench_v2.oneoffs.build_portraits \
        --csv results/token_scoring/qwen3_vl/easyportrait/prompt_token_fg_bg_stats_combined_ideology_headwise_linear.csv \
        --out bench_v2/tasks/s3_digest/portraits_v1.jsonl --per-bucket 250
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

BUCKETS = ("low", "mid", "high")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the EasyPortrait portrait pool.")
    p.add_argument("--csv", default="results/token_scoring/qwen3_vl/easyportrait/"
                                    "prompt_token_fg_bg_stats_combined_ideology_headwise_linear.csv")
    p.add_argument("--out", default="bench_v2/tasks/s3_digest/portraits_v1.jsonl")
    p.add_argument("--per-bucket", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        try:
            score = float(r["image_mean"])
        except (KeyError, TypeError, ValueError):
            continue
        scored.append((score, r))
    scored.sort(key=lambda t: t[0])

    n = len(scored)
    cuts = (n // 3, 2 * n // 3)
    ranked: list[tuple[str, float, dict[str, Any]]] = []
    for idx, (score, r) in enumerate(scored):
        bucket = "low" if idx < cuts[0] else ("mid" if idx < cuts[1] else "high")
        ranked.append((bucket, score, r))

    rng = random.Random(args.seed)
    pool: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        cand = [(s, r) for b, s, r in ranked if b == bucket]
        chosen = rng.sample(cand, min(args.per_bucket, len(cand)))
        chosen.sort(key=lambda t: t[0])
        for score, r in chosen:
            pool.append({
                "record_id": r["record_id"],
                "record_name": r["record_name"],
                "image_path": r["image_path"],
                "image_mean": round(float(score), 6),
                "bucket": bucket,
            })

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in pool:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()[:16]
    canonical = ("results/token_scoring/qwen3_vl/easyportrait/"
                 "prompt_token_fg_bg_stats_combined_ideology_headwise_linear.csv")
    meta = {
        "source": "EasyPortrait",
        "csv": canonical,
        "csv_sha256_16": digest,
        "probe": "combined_ideology_headwise_linear",
        "score_field": "image_mean",
        "bucketing": "rank_tertile",
        "buckets": list(BUCKETS),
        "n_total": n,
        "cuts": {"low_hi": round(scored[cuts[0] - 1][0], 6),
                 "mid_hi": round(scored[cuts[1] - 1][0], 6)},
        "per_bucket": args.per_bucket,
        "seed": args.seed,
        "note": ("Portrait buckets are rank tertiles because the portrait probe scores "
                 "do not straddle the fixed +/-0.5 cut the LVIS item buckets use."),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(pool)} portraits -> {out}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
