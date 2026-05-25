#!/usr/bin/env python3
"""
Add foreground/background image-token statistics to EasyPortrait token score CSVs.

Uses EasyPortrait segmentation masks to classify each image token as foreground
(person, skin, brows, eyes, lips, teeth: classes 1-8) or background (class 0),
then computes per-image statistics for each group.

Writes new CSV files named prompt_token_fg_bg_stats_*.csv -- original CSVs are
left untouched.

New columns:
  num_fg_tokens, num_bg_tokens, fg_ratio
  image_fg_mean, image_fg_median, image_fg_min, image_fg_max, image_fg_std
  image_bg_mean, image_bg_median, image_bg_min, image_bg_max, image_bg_std
  image_last_token_score
  all_last_token_score

Processes all models by default.
"""

import argparse
import csv
import multiprocessing as mp
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

DEFAULT_ANNOTATIONS_DIR = "datasets/EasyPortrait/data/annotations"
DEFAULT_RESULTS_DIR = "results/token_scoring"

NEW_FG_BG_COLUMNS = [
    "num_fg_tokens",
    "num_bg_tokens",
    "fg_ratio",
    "image_fg_mean",
    "image_fg_median",
    "image_fg_min",
    "image_fg_max",
    "image_fg_std",
    "image_bg_mean",
    "image_bg_median",
    "image_bg_min",
    "image_bg_max",
    "image_bg_std",
    "all_last_token_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add fg/bg token scores to EasyPortrait CSVs using segmentation masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations-dir",
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Directory containing EasyPortrait annotation masks (train/test/val subdirs).",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing per-model token scoring results.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to process (e.g. gemma4, qwen3_vl). If not set, process all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying any files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu count).",
    )
    return parser.parse_args()


def build_annotation_index(annotations_dir: str) -> Dict[str, str]:
    """Build a UUID-to-path mapping for all annotation PNGs across splits."""
    index: Dict[str, str] = {}
    for split in ("train", "test", "val"):
        split_dir = os.path.join(annotations_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if fname.endswith(".png"):
                index[fname[:-4]] = os.path.join(split_dir, fname)
    return index


def compute_fg_mask_block(args_tuple: Tuple[str, int, int]) -> Tuple[str, Optional[Dict]]:
    """
    Load a single annotation, compute fg/bg classification per grid cell.
    Returns (uuid, result_dict) or (uuid, None) on failure.
    """
    ann_path, grid_h, grid_w = args_tuple
    try:
        ann = np.array(Image.open(ann_path))
    except Exception:
        return (ann_path, None)

    fg = (ann > 0).astype(np.float32)
    ah, aw = ann.shape

    # Pad so that image dimensions are divisible by grid dimensions
    pad_h = (grid_h - (ah % grid_h)) % grid_h
    pad_w = (grid_w - (aw % grid_w)) % grid_w
    if pad_h > 0 or pad_w > 0:
        fg = np.pad(fg, ((0, pad_h), (0, pad_w)), mode="edge")

    block_h = fg.shape[0] // grid_h
    block_w = fg.shape[1] // grid_w

    fg_blocks = fg.reshape(grid_h, block_h, grid_w, block_w)
    fg_proportions = fg_blocks.mean(axis=(1, 3))

    num_fg = int((fg_proportions > 0.5).sum())
    num_bg = int(grid_h * grid_w) - num_fg

    uuid = os.path.basename(ann_path)[:-4]
    return (uuid, {
        "num_fg": num_fg,
        "num_bg": num_bg,
        "fg_ratio": num_fg / (grid_h * grid_w) if grid_h * grid_w > 0 else float("nan"),
        "fg_mask": fg_proportions > 0.5,  # shape: (grid_h, grid_w), bool
    })


def uuid_from_name(record_name: str) -> str:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"):
        if record_name.endswith(ext):
            return record_name[: -len(ext)]
    return record_name


def find_annotation_path(uuid: str, ann_index: Dict[str, str], annotations_dir: str) -> Optional[str]:
    path = ann_index.get(uuid)
    if path is not None:
        return path
    for split in ("train", "test", "val"):
        candidate = os.path.join(annotations_dir, split, f"{uuid}.png")
        if os.path.exists(candidate):
            return candidate
    return None


def build_fg_cache(
    record_rows: List[Dict[str, str]],
    ann_index: Dict[str, str],
    annotations_dir: str,
    num_workers: int,
) -> Dict[str, Dict]:
    """
    Compute fg/bg classification for all records in parallel, using per-record grid dims.
    Returns: {record_id: {"num_fg": int, "num_bg": int, "fg_ratio": float, "fg_mask": ndarray}}
    """
    tasks: List[Tuple[str, int, int]] = []
    uuid_to_rid: Dict[str, str] = {}
    for row in record_rows:
        uuid = uuid_from_name(row["record_name"])
        ann_path = find_annotation_path(uuid, ann_index, annotations_dir)
        if ann_path is None:
            continue
        grid_h = int(row["grid_h"])
        grid_w = int(row["grid_w"])
        tasks.append((ann_path, grid_h, grid_w))
        uuid_to_rid[uuid] = row["record_id"]

    if not tasks:
        return {}

    n_workers = num_workers or min(mp.cpu_count() or 1, len(tasks))
    print(f"  Computing fg/bg masks for {len(tasks)} annotations ({n_workers} workers)...")

    with mp.Pool(n_workers) as pool:
        results = list(pool.imap_unordered(compute_fg_mask_block, tasks, chunksize=200))

    cache: Dict[str, Dict] = {}
    for uuid, data in results:
        if data is not None and uuid in uuid_to_rid:
            cache[uuid_to_rid[uuid]] = data

    # Fallback: any records that were in ann_index but missed in results
    for row in record_rows:
        record_id = row["record_id"]
        if record_id in cache:
            continue
        uuid = uuid_from_name(row["record_name"])
        ann_path = find_annotation_path(uuid, ann_index, annotations_dir)
        if ann_path is not None:
            gh = int(row["grid_h"])
            gw = int(row["grid_w"])
            _, data = compute_fg_mask_block((ann_path, gh, gw))
            if data is not None:
                cache[record_id] = data

    return cache


def summarize(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def update_csv_with_cache(
    csv_path: str,
    npz_path: str,
    all_npz_path: str,
    fg_cache: Dict[str, Dict],
    dry_run: bool = False,
) -> int:
    """Read original CSV + NPZs, apply cached fg/bg classification, write NEW CSV."""

    out_path = csv_path.replace("token_stats", "token_fg_bg_stats")

    npz_data = np.load(npz_path, allow_pickle=False)
    all_npz_data = np.load(all_npz_path, allow_pickle=False)

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        original_fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    if not rows:
        return 0

    updated = 0
    skipped = 0

    for row in rows:
        record_id = row["record_id"]
        cache_entry = fg_cache.get(record_id)

        if cache_entry is None:
            skipped += 1
            for col in NEW_FG_BG_COLUMNS:
                row[col] = ""
            continue

        if record_id not in npz_data:
            skipped += 1
            for col in NEW_FG_BG_COLUMNS:
                row[col] = ""
            continue

        token_scores = npz_data[record_id]
        fg_mask = cache_entry["fg_mask"].flatten()

        n_tokens = min(len(fg_mask), len(token_scores))
        fg_mask = fg_mask[:n_tokens]
        token_scores = token_scores[:n_tokens]

        fg_scores = token_scores[fg_mask]
        bg_scores = token_scores[~fg_mask]

        fg_stats = summarize(fg_scores)
        bg_stats = summarize(bg_scores)

        # Last token score from the all-tokens (full prompt) NPZ
        if record_id in all_npz_data:
            all_scores = all_npz_data[record_id]
            last_score = float(all_scores[-1]) if len(all_scores) > 0 else float("nan")
        else:
            last_score = float("nan")

        row["num_fg_tokens"] = str(cache_entry["num_fg"])
        row["num_bg_tokens"] = str(cache_entry["num_bg"])
        row["fg_ratio"] = f"{cache_entry['fg_ratio']:.6f}"
        row["image_fg_mean"] = str(fg_stats["mean"])
        row["image_fg_median"] = str(fg_stats["median"])
        row["image_fg_min"] = str(fg_stats["min"])
        row["image_fg_max"] = str(fg_stats["max"])
        row["image_fg_std"] = str(fg_stats["std"])
        row["image_bg_mean"] = str(bg_stats["mean"])
        row["image_bg_median"] = str(bg_stats["median"])
        row["image_bg_min"] = str(bg_stats["min"])
        row["image_bg_max"] = str(bg_stats["max"])
        row["image_bg_std"] = str(bg_stats["std"])
        row["all_last_token_score"] = str(last_score)

        updated += 1

    if dry_run:
        print(f"  [DRY-RUN] {os.path.basename(csv_path)} -> {os.path.basename(out_path)} ({updated} rows)")
        return updated

    if updated > 0:
        output_fieldnames = original_fieldnames + NEW_FG_BG_COLUMNS
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {os.path.basename(out_path)}: {updated} rows")

    if skipped:
        print(f"  {os.path.basename(csv_path)}: skipped {skipped} rows (no annotation/NPZ data)")

    return updated


def find_csv_npz_pairs(easyportrait_dir: str) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    for fname in sorted(os.listdir(easyportrait_dir)):
        if not fname.endswith(".csv"):
            continue
        csv_path = os.path.join(easyportrait_dir, fname)
        img_npz_name = fname.replace("token_stats", "image_token_scores").replace(".csv", ".npz")
        img_npz_path = os.path.join(easyportrait_dir, img_npz_name)
        all_npz_name = fname.replace("token_stats", "all_token_scores").replace(".csv", ".npz")
        all_npz_path = os.path.join(easyportrait_dir, all_npz_name)
        if os.path.exists(img_npz_path) and os.path.exists(all_npz_path):
            pairs.append((csv_path, img_npz_path, all_npz_path))
    return pairs


def main() -> None:
    args = parse_args()

    ann_index = build_annotation_index(args.annotations_dir)
    print(f"Annotation index: {len(ann_index)} entries")

    models = (
        [args.model]
        if args.model
        else sorted(
            d
            for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d))
        )
    )

    for model in models:
        model_dir = os.path.join(args.results_dir, model)
        ep_dir = os.path.join(model_dir, "easyportrait")
        if not os.path.isdir(ep_dir):
            continue

        pairs = find_csv_npz_pairs(ep_dir)
        if not pairs:
            continue

        # Read all rows from first CSV (grid dims vary per image but are identical
        # across all CSVs for the same model, so we build the cache from one CSV).
        first_csv = pairs[0][0]
        with open(first_csv, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows_for_cache = list(reader)

        unique_grids = sorted(set((r["grid_h"], r["grid_w"]) for r in rows_for_cache))
        grid_summary = ", ".join(f"{h}x{w}" for h, w in unique_grids[:5])
        if len(unique_grids) > 5:
            grid_summary += f", ... ({len(unique_grids)} total)"

        print(f"\n{'='*60}")
        print(f"Model: {model}  (grids: {grid_summary})")
        print(f"{'='*60}")

        # Compute fg/bg cache once for all CSVs (per-record grid dims)
        fg_cache = build_fg_cache(
            rows_for_cache,
            ann_index,
            annotations_dir=args.annotations_dir,
            num_workers=args.workers,
        )
        print(f"  Cached fg/bg data for {len(fg_cache)} records")

        for csv_path, img_npz_path, all_npz_path in pairs:
            update_csv_with_cache(csv_path, img_npz_path, all_npz_path, fg_cache, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
