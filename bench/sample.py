"""Sampler: LVIS score CSV -> frozen stimulus items.

**Integrity filters only, by default** (task F). ``image_token_mismatch == 0``
and ``num_image_tokens > 0`` are about whether a row is usable data at all;
everything else that used to be a filter -- no_person, no_text_cats, objects:,
aspect:, tokens: -- is still implemented and still reachable through
``--filters``, but it is off by default and belongs to sensitivity analysis.

The reason is not that those filters were badly tuned (``tokens:300-800`` was --
it cut 58% of a pool whose maximum is 400). It is that dropping images by content
conditions on variables that may sit on the causal path, which controls away the
mechanism, and every threshold is a researcher degree of freedom. So the same
quantities are carried as *annotation columns* on each item instead:
``n_persons``, ``has_text_cat``, ``n_objects``, ``aspect``, ``num_image_tokens``.

Then: stratify on ``image_mean`` into deciles, balance ``n_objects`` within a
stratum, group images into items of 3 *from the same stratum*, cut explore/confirm.
The stratum index is the primary independent variable (task E).
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_STATS_CSV = os.path.join(
    "results", "token_scoring", "qwen3_vl", "lvis",
    "prompt_token_stats_combined_ideology_headwise_linear.csv",
)
DEFAULT_LVIS_JSON = os.path.join("datasets", "lvis", "lvis_v1_train.json")
# v2: the cache now carries per-category instance counts, which n_persons needs.
DEFAULT_LVIS_CACHE = os.path.join("items", "_cache", "lvis_image_meta_v2.jsonl")

# LVIS categories that put readable text in the frame (docs/bench/03). Now an
# annotation (`has_text_cat`) rather than a filter.
TEXT_CATEGORIES = frozenset({
    "signboard", "street_sign", "poster", "flag", "book", "newspaper",
    "license_plate", "magazine",
})
PERSON_CATEGORIES = frozenset({"person"})

# Integrity only. The content filters below are opt-in via --filters.
DEFAULT_FILTERS = ""
SENSITIVITY_FILTERS = "no_person,no_text_cats,objects:3-15,aspect:0.6-1.7,tokens:64-400"

csv.field_size_limit(10_000_000)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_stats(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "record_id": raw["record_id"],
                "record_name": raw["record_name"],
                "image_path": raw["image_path"],
                "image_h": int(raw["image_h"]),
                "image_w": int(raw["image_w"]),
                "num_image_tokens": int(raw["num_image_tokens"]),
                "image_token_mismatch": int(raw["image_token_mismatch"]),
                "image_mean": float(raw["image_mean"]),
                "coco_id": coco_id_from_record_name(raw["record_name"]),
            })
            if limit and len(rows) >= limit:
                break
    return rows


def coco_id_from_record_name(record_name: str) -> int:
    """'train2017/000000000030.jpg' -> 30"""
    stem = os.path.splitext(os.path.basename(record_name))[0]
    return int(stem)


def build_lvis_meta_cache(lvis_json: str, cache_path: str) -> str:
    """Reduce the 1.1 GB annotation file to one compact line per image.

    Written once; every later `bench sample` reads the cache instead.
    """
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    print(f"[sample] reading {lvis_json} (this takes a minute and a lot of RAM)", flush=True)
    with open(lvis_json, encoding="utf-8") as handle:
        lvis = json.load(handle)

    cat_name = {c["id"]: c["name"] for c in lvis["categories"]}
    per_image: Dict[int, Counter] = defaultdict(Counter)
    for ann in lvis["annotations"]:
        per_image[ann["image_id"]][cat_name.get(ann["category_id"], "?")] += 1

    known_images = {img["id"] for img in lvis["images"]}
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for image_id in sorted(known_images):
            counts = per_image.get(image_id, Counter())
            handle.write(json.dumps({
                "coco_id": image_id,
                "n_objects": int(sum(counts.values())),
                "categories": sorted(counts),
                # instance counts, not just presence: n_persons needs the count
                "category_counts": {k: int(v) for k, v in sorted(counts.items())},
            }, sort_keys=True) + "\n")
    os.replace(tmp, cache_path)
    print(f"[sample] wrote {cache_path} ({len(known_images)} images)", flush=True)
    return cache_path


def load_lvis_meta(cache_path: str) -> Dict[int, Dict[str, Any]]:
    meta: Dict[int, Dict[str, Any]] = {}
    with open(cache_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts = row.get("category_counts") or {c: 1 for c in row["categories"]}
            meta[int(row["coco_id"])] = {
                "n_objects": int(row["n_objects"]),
                "categories": list(row["categories"]),
                "category_counts": {k: int(v) for k, v in counts.items()},
            }
    return meta


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
def parse_filters(spec: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "no_person": False, "no_text_cats": False,
        "objects": None, "aspect": None, "tokens": None,
    }
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        if token in ("no_person", "no_text_cats"):
            out[token] = True
        elif ":" in token:
            key, rng = token.split(":", 1)
            lo, hi = rng.split("-", 1)
            if key not in ("objects", "aspect", "tokens"):
                raise ValueError(f"Unknown filter {key!r}")
            out[key] = (float(lo), float(hi))
        else:
            raise ValueError(f"Unknown filter {token!r}")
    return out


def apply_filters(
    rows: Sequence[Dict[str, Any]],
    meta: Dict[int, Dict[str, Any]],
    filters: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Counter]:
    kept: List[Dict[str, Any]] = []
    dropped: Counter = Counter()

    for row in rows:
        # Integrity, always on: these two say the row is not usable data at all.
        if row["image_token_mismatch"] != 0:
            dropped["integrity_token_mismatch"] += 1
            continue
        if row["num_image_tokens"] <= 0:
            dropped["integrity_no_image_tokens"] += 1
            continue

        # Everything below is opt-in via --filters (task F): off by default,
        # kept because they are the sensitivity analyses.
        if filters["tokens"]:
            lo, hi = filters["tokens"]
            if not (lo <= row["num_image_tokens"] <= hi):
                dropped["opt_token_count"] += 1
                continue

        info = meta.get(row["coco_id"])
        if info is None:
            dropped["no_lvis_annotation"] += 1
            continue
        cats = set(info["categories"])

        if filters["no_person"] and (cats & PERSON_CATEGORIES):
            dropped["opt_person"] += 1
            continue
        if filters["no_text_cats"] and (cats & TEXT_CATEGORIES):
            dropped["opt_text_category"] += 1
            continue
        if filters["objects"]:
            lo, hi = filters["objects"]
            if not (lo <= info["n_objects"] <= hi):
                dropped["opt_n_objects"] += 1
                continue
        if filters["aspect"]:
            lo, hi = filters["aspect"]
            aspect = row["image_w"] / row["image_h"] if row["image_h"] else 0.0
            if not (lo <= aspect <= hi):
                dropped["opt_aspect"] += 1
                continue

        enriched = dict(row)
        enriched["n_objects"] = info["n_objects"]
        enriched["categories"] = info["categories"]
        enriched["aspect"] = row["image_w"] / row["image_h"] if row["image_h"] else 0.0
        # Annotations that used to be filters (task F).
        # Caveat that has to travel with n_persons: LVIS is *federated* -- each
        # category is exhaustively annotated only in the images of its own subset,
        # and `person` is annotated in 1,928 of 100,170 images (1.9%). So
        # n_persons == 0 means "no person annotation", not "no person in frame".
        # The old no_person filter therefore never did what it claimed either.
        enriched["n_persons"] = int(info.get("category_counts", {}).get("person", 0))
        enriched["has_text_cat"] = bool(cats & TEXT_CATEGORIES)
        enriched["text_cats"] = sorted(cats & TEXT_CATEGORIES)
        kept.append(enriched)

    return kept, dropped


# --------------------------------------------------------------------------- #
# stratification
# --------------------------------------------------------------------------- #
def assign_strata(rows: Sequence[Dict[str, Any]], bins: int = 10) -> List[Dict[str, Any]]:
    """Equal-count bins on image_mean: stratum 0 is most left, bins-1 most right.

    The stratum index is the primary IV (task E): the three images of an item are
    drawn from one stratum, so the stratum -- not the mean of three separately
    measured scores -- is what was actually manipulated.
    """
    ordered = sorted(rows, key=lambda r: (r["image_mean"], r["record_id"]))
    n = len(ordered)
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(ordered):
        row = dict(row)
        row["stratum"] = min(bins - 1, (index * bins) // n) if n else 0
        out.append(row)
    return out


assign_deciles = assign_strata   # v1 name, kept so old imports still resolve


# --------------------------------------------------------------------------- #
# buckets (round 4): three fixed thresholds on image_mean, not equal-count bins
# --------------------------------------------------------------------------- #
# Thresholds sit on the probe's zero point (±0.5), not on the corpus quantiles
# (board-buckets): 0 is the DW-NOMINATE midpoint, so the bucket boundaries mean
# the same thing regardless of how the corpus happens to be distributed.
BUCKETS = ("low", "mid", "high")
BUCKET_ORDINAL = {"low": -1, "mid": 0, "high": 1}
BUCKET_ABBREV = {"low": "lo", "mid": "mid", "high": "hi"}
BUCKET_LO = -0.5
BUCKET_HI = 0.5


def bucket_of(image_mean: float) -> str:
    if image_mean < BUCKET_LO:
        return "low"
    if image_mean <= BUCKET_HI:
        return "mid"
    return "high"


def assign_buckets(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tag every row with its bucket name and the ordinal (-1/0/+1) that carries
    the row's `stratum` slot. `image_mean` itself is untouched: the bucket only
    guarantees the two tails are covered; the main analysis stays continuous."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        bucket = bucket_of(row["image_mean"])
        row["bucket"] = bucket
        row["stratum"] = BUCKET_ORDINAL[bucket]
        out.append(row)
    return out


def _objects_bucket(n_objects: int) -> str:
    if n_objects <= 4:
        return "3-4"
    if n_objects <= 7:
        return "5-7"
    if n_objects <= 10:
        return "8-10"
    return "11-15"


def _category_count(row: Dict[str, Any]) -> int:
    """The board's ``n_objects`` (board-buckets: 3.9 / 3.7 / 2.7 across buckets) is
    the number of *distinct* LVIS categories, not the cache's ``n_objects`` field
    (which is the instance count, ~13.7 / 13.0 / 10.2). Scene complexity here is
    "how many different kinds of thing are in the frame", so that is what the
    balance targets."""
    return len(row.get("categories") or [])


def balanced_pick(
    candidates: Sequence[Dict[str, Any]],
    n: int,
    target_shares: Dict[str, float],
    rng: random.Random,
    key: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> List[Dict[str, Any]]:
    """Pick n rows matching the pool-wide n_objects mix as closely as possible.

    ``key`` maps a row to its balancing bucket. The decile path (``key=None``)
    balances the cache's ``n_objects`` field exactly as before; the bucket path
    passes ``_category_count`` so it balances distinct-category count instead.
    """
    key = key or (lambda row: _objects_bucket(row["n_objects"]))
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bucket[key(row)].append(row)
    for bucket in by_bucket.values():
        rng.shuffle(bucket)

    picked: List[Dict[str, Any]] = []
    for bucket, share in sorted(target_shares.items()):
        want = int(round(share * n))
        take = by_bucket[bucket][:want]
        by_bucket[bucket] = by_bucket[bucket][want:]
        picked.extend(take)

    if len(picked) < n:  # top up from whatever is left
        leftovers = [row for bucket in by_bucket.values() for row in bucket]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: n - len(picked)])
    return picked[:n]


def make_items(
    rows: Sequence[Dict[str, Any]],
    bins: int = 10,
    per_bin: int = 400,
    images_per_item: int = 3,
    splits: Sequence[str] = ("explore", "confirm"),
    seed: int = 42,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Return {split: [item, ...]} plus a profile dict.

    ``per_bin`` counts *images* per stratum; items_per_bin = per_bin // images_per_item.
    All images of an item come from the same stratum.
    """
    rng = random.Random(seed)
    rows = assign_strata(rows, bins=bins)

    global_counts = Counter(_objects_bucket(r["n_objects"]) for r in rows)
    total = sum(global_counts.values()) or 1
    target_shares = {b: c / total for b, c in global_counts.items()}

    by_stratum: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[row["stratum"]].append(row)

    items_per_bin = per_bin // images_per_item
    per_split = items_per_bin // len(splits)
    if per_split == 0:
        raise ValueError(
            f"per_bin={per_bin} with images_per_item={images_per_item} and {len(splits)} splits "
            f"gives 0 items per split; raise --per-bin"
        )

    out: Dict[str, List[Dict[str, Any]]] = {split: [] for split in splits}
    profile_rows: List[Dict[str, Any]] = []
    counter = 0

    for stratum in range(bins):
        pool = by_stratum.get(stratum, [])
        need = per_split * len(splits) * images_per_item
        chosen = balanced_pick(pool, min(need, len(pool)), target_shares, rng)
        rng.shuffle(chosen)

        groups = [chosen[i: i + images_per_item]
                  for i in range(0, len(chosen) - images_per_item + 1, images_per_item)]
        groups = groups[: per_split * len(splits)]

        for group_index, group in enumerate(groups):
            split = splits[group_index % len(splits)]
            scores = [row["image_mean"] for row in group]
            cats = sorted({c for row in group for c in row["categories"]})
            item = {
                "item_id": f"lvis{images_per_item}_{counter:05d}",
                "images": [row["record_name"] for row in group],
                "image_paths": [row["image_path"] for row in group],
                "image_scores": scores,
                "stratum": stratum,
                # Explicit, because image_mean_mean is *not* the x axis (task E):
                # three images read together do not score like the mean of three
                # images read alone, and the asymmetry is direction-dependent.
                "primary_iv": "stratum",
                "split": split,
                "covariates": {
                    "coco_ids": [row["coco_id"] for row in group],
                    "n_objects": [row["n_objects"] for row in group],
                    "n_objects_mean": sum(row["n_objects"] for row in group) / len(group),
                    "n_persons": [row["n_persons"] for row in group],
                    "n_persons_total": sum(row["n_persons"] for row in group),
                    "has_text_cat": [row["has_text_cat"] for row in group],
                    "has_text_cat_any": any(row["has_text_cat"] for row in group),
                    "text_cats": sorted({c for row in group for c in row["text_cats"]}),
                    "num_image_tokens": [row["num_image_tokens"] for row in group],
                    "aspect": [round(row["aspect"], 4) for row in group],
                    "categories": cats,
                    # a covariate now, not the independent variable
                    "image_mean_mean": sum(scores) / len(scores),
                },
            }
            out[split].append(item)
            counter += 1

        if chosen:
            top_cats = Counter(c for row in chosen for c in row["categories"]).most_common(10)
            profile_rows.append({
                "stratum": stratum,
                "n_images_selected": len(chosen),
                "image_mean_min": min(r["image_mean"] for r in chosen),
                "image_mean_max": max(r["image_mean"] for r in chosen),
                "image_mean_mean": sum(r["image_mean"] for r in chosen) / len(chosen),
                "n_objects_median": _median([r["n_objects"] for r in chosen]),
                "n_persons_mean": sum(r["n_persons"] for r in chosen) / len(chosen),
                "share_with_person": sum(1 for r in chosen if r["n_persons"] > 0) / len(chosen),
                "share_with_text_cat": sum(1 for r in chosen if r["has_text_cat"]) / len(chosen),
                "top_categories": top_cats,
            })

    profile = {
        "bins": bins, "per_bin": per_bin, "images_per_item": images_per_item,
        "seed": seed, "splits": list(splits), "primary_iv": "stratum",
        "n_items": {split: len(items) for split, items in out.items()},
        "objects_target_shares": target_shares,
        "strata": profile_rows,
    }
    return out, profile


def make_bucket_items(
    rows: Sequence[Dict[str, Any]],
    per_bucket: int = 400,
    images_per_item: int = 3,
    splits: Sequence[str] = ("explore", "confirm"),
    seed: int = 42,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Return {split: [item, ...]} plus a profile, stratified by bucket (round 4).

    Three images per item, all from the same bucket. ``per_bucket`` counts images
    per bucket; ``n_objects`` (distinct LVIS category count, board-buckets 3.9/3.7/2.7)
    is balanced *within* the bucket exactly as the decile path balances within a
    stratum, so ``bucket`` cannot collapse into ``scene complexity``. The item_id
    carries the bucket name (``lvis3_lo_00000`` / ``lvis3_mid_...`` / ``lvis3_hi_...``)
    and ``primary_iv`` is ``bucket`` (ordinal -1/0/+1); ``image_mean`` stays on the
    record as the continuous quantity for the main analysis.
    """
    rng = random.Random(seed)
    rows = assign_buckets(rows)

    global_counts = Counter(_objects_bucket(_category_count(r)) for r in rows)
    total = sum(global_counts.values()) or 1
    target_shares = {b: c / total for b, c in global_counts.items()}

    by_bucket: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BUCKETS}
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    items_per_bucket = per_bucket // images_per_item
    per_split = items_per_bucket // len(splits)
    if per_split == 0:
        raise ValueError(
            f"per_bucket={per_bucket} with images_per_item={images_per_item} and {len(splits)} "
            f"splits gives 0 items per split; raise --per-bucket"
        )

    out: Dict[str, List[Dict[str, Any]]] = {split: [] for split in splits}
    profile_rows: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {b: 0 for b in BUCKETS}

    for bucket in BUCKETS:
        pool = by_bucket.get(bucket, [])
        need = per_split * len(splits) * images_per_item
        chosen = balanced_pick(pool, min(need, len(pool)), target_shares, rng,
                               key=_category_count)
        rng.shuffle(chosen)

        groups = [chosen[i: i + images_per_item]
                  for i in range(0, len(chosen) - images_per_item + 1, images_per_item)]
        groups = groups[: per_split * len(splits)]

        for group_index, group in enumerate(groups):
            split = splits[group_index % len(splits)]
            scores = [row["image_mean"] for row in group]
            cats = sorted({c for row in group for c in row["categories"]})
            n_objects = [row["n_objects"] for row in group]
            n_categories = [len(row["categories"]) for row in group]
            item = {
                # The image count is in the id because item_id is part of
                # `trial_key`: a 10-image item named lvis3_* would dedup against
                # an existing 3-image trial and be silently skipped.
                "item_id": f"lvis{images_per_item}_{BUCKET_ABBREV[bucket]}_{counters[bucket]:05d}",
                "images": [row["record_name"] for row in group],
                "image_paths": [row["image_path"] for row in group],
                "image_scores": scores,
                "stratum": BUCKET_ORDINAL[bucket],
                "bucket": bucket,
                "primary_iv": "bucket",
                "split": split,
                "covariates": {
                    "bucket": bucket,
                    "coco_ids": [row["coco_id"] for row in group],
                    "n_objects": n_objects,
                    "n_objects_mean": sum(n_objects) / len(n_objects),
                    "n_categories": n_categories,
                    "n_categories_mean": sum(n_categories) / len(n_categories),
                    "n_persons": [row["n_persons"] for row in group],
                    "n_persons_total": sum(row["n_persons"] for row in group),
                    "has_text_cat": [row["has_text_cat"] for row in group],
                    "has_text_cat_any": any(row["has_text_cat"] for row in group),
                    "text_cats": sorted({c for row in group for c in row["text_cats"]}),
                    "num_image_tokens": [row["num_image_tokens"] for row in group],
                    "aspect": [round(row["aspect"], 4) for row in group],
                    "categories": cats,
                    "image_mean_mean": sum(scores) / len(scores),
                },
            }
            out[split].append(item)
            counters[bucket] += 1

        if chosen:
            top_cats = Counter(c for row in chosen for c in row["categories"]).most_common(10)
            profile_rows.append({
                "bucket": bucket,
                "n_images_available": len(pool),
                "n_images_selected": len(chosen),
                "image_mean_min": min(r["image_mean"] for r in chosen),
                "image_mean_max": max(r["image_mean"] for r in chosen),
                "image_mean_mean": sum(r["image_mean"] for r in chosen) / len(chosen),
                "n_objects_median": _median([r["n_objects"] for r in chosen]),
                "n_categories_mean": sum(_category_count(r) for r in chosen) / len(chosen),
                "n_persons_mean": sum(r["n_persons"] for r in chosen) / len(chosen),
                "share_with_person": sum(1 for r in chosen if r["n_persons"] > 0) / len(chosen),
                "share_with_text_cat": sum(1 for r in chosen if r["has_text_cat"]) / len(chosen),
                "top_categories": top_cats,
            })

    profile = {
        "stratification": "buckets", "per_bucket": per_bucket,
        "images_per_item": images_per_item, "seed": seed,
        "splits": list(splits), "primary_iv": "bucket",
        "n_items": {split: len(items) for split, items in out.items()},
        "objects_target_shares": target_shares,
        "buckets": profile_rows,
    }
    return out, profile


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    return float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def write_decile_profile(profile: Dict[str, Any], path: str) -> str:
    """Per-stratum frequent categories + the annotation shares.

    The category frequencies are reported, never used to select (docs/bench/03:
    "别为了显得干净去砍掉那些类别"). No indoor/outdoor mapping is invented here --
    that would be one more subjective intervention; the counts are printed and
    the reader judges.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stratum", "n_images_selected", "image_mean_min", "image_mean_mean",
                         "image_mean_max", "n_objects_median", "n_persons_mean",
                         "share_with_person", "share_with_text_cat", "top_categories"])
        for row in profile["strata"]:
            writer.writerow([
                row["stratum"], row["n_images_selected"],
                f"{row['image_mean_min']:.4f}", f"{row['image_mean_mean']:.4f}",
                f"{row['image_mean_max']:.4f}", row["n_objects_median"],
                f"{row['n_persons_mean']:.3f}", f"{row['share_with_person']:.3f}",
                f"{row['share_with_text_cat']:.3f}",
                ";".join(f"{name}:{count}" for name, count in row["top_categories"]),
            ])
    return path


def write_bucket_profile(profile: Dict[str, Any], path: str) -> str:
    """Per-bucket frequent categories + annotation shares (round 4).

    Same reporting-only semantics as ``write_decile_profile``: the category
    frequencies are printed, never used to select. ``n_categories_mean`` is the
    post-balance number the board's 3.9/3.7/2.7 self-check is about.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "n_images_available", "n_images_selected",
                         "image_mean_min", "image_mean_mean", "image_mean_max",
                         "n_objects_median", "n_categories_mean", "n_persons_mean",
                         "share_with_person", "share_with_text_cat", "top_categories"])
        for row in profile["buckets"]:
            writer.writerow([
                row["bucket"], row["n_images_available"], row["n_images_selected"],
                f"{row['image_mean_min']:.4f}", f"{row['image_mean_mean']:.4f}",
                f"{row['image_mean_max']:.4f}", row["n_objects_median"],
                f"{row['n_categories_mean']:.4f}",
                f"{row['n_persons_mean']:.3f}", f"{row['share_with_person']:.3f}",
                f"{row['share_with_text_cat']:.3f}",
                ";".join(f"{name}:{count}" for name, count in row["top_categories"]),
            ])
    return path
