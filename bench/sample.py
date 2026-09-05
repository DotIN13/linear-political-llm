"""Sampler: LVIS score CSV -> frozen stimulus items.

Filters F1-F3 and F5-F6 from docs/bench/03, then decile-stratifies on
``image_mean``, balances ``n_objects`` inside each stratum, groups images into
items of 3 *from the same decile* (so the item mean does not wash the extreme
out), and cuts explore/confirm.

F4 (OCR against a political wordlist) is NOT implemented here -- see bench/README.md.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_STATS_CSV = os.path.join(
    "results", "token_scoring", "qwen3_vl", "lvis",
    "prompt_token_stats_combined_ideology_headwise_linear.csv",
)
DEFAULT_LVIS_JSON = os.path.join("datasets", "lvis", "lvis_v1_train.json")
DEFAULT_LVIS_CACHE = os.path.join("items", "_cache", "lvis_image_meta.jsonl")

# F3: LVIS categories that put readable text in the frame (docs/bench/03).
TEXT_CATEGORIES = frozenset({
    "signboard", "street_sign", "poster", "flag", "book", "newspaper",
    "license_plate", "magazine",
})
PERSON_CATEGORIES = frozenset({"person"})

DEFAULT_FILTERS = "no_person,no_text_cats,objects:3-15,aspect:0.6-1.7,tokens:300-800"

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
    per_image: Dict[int, List[str]] = defaultdict(list)
    for ann in lvis["annotations"]:
        per_image[ann["image_id"]].append(cat_name.get(ann["category_id"], "?"))

    known_images = {img["id"] for img in lvis["images"]}
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for image_id in sorted(known_images):
            cats = per_image.get(image_id, [])
            handle.write(json.dumps({
                "coco_id": image_id,
                "n_objects": len(cats),
                "categories": sorted(set(cats)),
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
            meta[int(row["coco_id"])] = {
                "n_objects": int(row["n_objects"]),
                "categories": list(row["categories"]),
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
        # F1 data quality
        if row["image_token_mismatch"] != 0:
            dropped["F1_token_mismatch"] += 1
            continue
        if filters["tokens"]:
            lo, hi = filters["tokens"]
            if not (lo <= row["num_image_tokens"] <= hi):
                dropped["F1_token_count"] += 1
                continue

        info = meta.get(row["coco_id"])
        if info is None:
            dropped["no_lvis_annotation"] += 1
            continue
        cats = set(info["categories"])

        if filters["no_person"] and (cats & PERSON_CATEGORIES):
            dropped["F2_person"] += 1
            continue
        if filters["no_text_cats"] and (cats & TEXT_CATEGORIES):
            dropped["F3_text_category"] += 1
            continue
        if filters["objects"]:
            lo, hi = filters["objects"]
            if not (lo <= info["n_objects"] <= hi):
                dropped["F5_n_objects"] += 1
                continue
        if filters["aspect"]:
            lo, hi = filters["aspect"]
            aspect = row["image_w"] / row["image_h"] if row["image_h"] else 0.0
            if not (lo <= aspect <= hi):
                dropped["F6_aspect"] += 1
                continue

        enriched = dict(row)
        enriched["n_objects"] = info["n_objects"]
        enriched["categories"] = info["categories"]
        enriched["aspect"] = row["image_w"] / row["image_h"] if row["image_h"] else 0.0
        kept.append(enriched)

    return kept, dropped


# --------------------------------------------------------------------------- #
# stratification
# --------------------------------------------------------------------------- #
def assign_deciles(rows: Sequence[Dict[str, Any]], bins: int = 10) -> List[Dict[str, Any]]:
    """Equal-count bins on image_mean: decile 0 is most left, bins-1 most right."""
    ordered = sorted(rows, key=lambda r: (r["image_mean"], r["record_id"]))
    n = len(ordered)
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(ordered):
        row = dict(row)
        row["decile"] = min(bins - 1, (index * bins) // n) if n else 0
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


def balanced_pick(
    candidates: Sequence[Dict[str, Any]],
    n: int,
    target_shares: Dict[str, float],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Pick n rows matching the pool-wide n_objects mix as closely as possible."""
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bucket[_objects_bucket(row["n_objects"])].append(row)
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

    ``per_bin`` counts *images* per decile; items_per_bin = per_bin // images_per_item.
    All images of an item come from the same decile.
    """
    rng = random.Random(seed)
    rows = assign_deciles(rows, bins=bins)

    global_counts = Counter(_objects_bucket(r["n_objects"]) for r in rows)
    total = sum(global_counts.values()) or 1
    target_shares = {b: c / total for b, c in global_counts.items()}

    by_decile: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_decile[row["decile"]].append(row)

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

    for decile in range(bins):
        pool = by_decile.get(decile, [])
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
                "decile": decile,
                "split": split,
                "covariates": {
                    "coco_ids": [row["coco_id"] for row in group],
                    "n_objects": [row["n_objects"] for row in group],
                    "n_objects_mean": sum(row["n_objects"] for row in group) / len(group),
                    "num_image_tokens": [row["num_image_tokens"] for row in group],
                    "aspect": [round(row["aspect"], 4) for row in group],
                    "categories": cats,
                    "image_mean_mean": sum(scores) / len(scores),
                },
            }
            out[split].append(item)
            counter += 1

        if chosen:
            top_cats = Counter(c for row in chosen for c in row["categories"]).most_common(10)
            profile_rows.append({
                "decile": decile,
                "n_images_selected": len(chosen),
                "image_mean_min": min(r["image_mean"] for r in chosen),
                "image_mean_max": max(r["image_mean"] for r in chosen),
                "image_mean_mean": sum(r["image_mean"] for r in chosen) / len(chosen),
                "n_objects_median": _median([r["n_objects"] for r in chosen]),
                "top_categories": top_cats,
            })

    profile = {
        "bins": bins, "per_bin": per_bin, "images_per_item": images_per_item,
        "seed": seed, "splits": list(splits),
        "n_items": {split: len(items) for split, items in out.items()},
        "objects_target_shares": target_shares,
        "deciles": profile_rows,
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
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["decile", "n_images_selected", "image_mean_min", "image_mean_mean",
                         "image_mean_max", "n_objects_median", "top_categories"])
        for row in profile["deciles"]:
            writer.writerow([
                row["decile"], row["n_images_selected"],
                f"{row['image_mean_min']:.4f}", f"{row['image_mean_mean']:.4f}",
                f"{row['image_mean_max']:.4f}", row["n_objects_median"],
                ";".join(f"{name}:{count}" for name, count in row["top_categories"]),
            ])
    return path
