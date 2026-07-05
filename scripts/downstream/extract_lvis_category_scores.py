#!/usr/bin/env python3
"""
Extract per-category political ideology scores from LVIS token scoring results.

Loads the per-token probe scores (.pt), per-image CSV stats, and LVIS annotations,
maps bounding boxes to vision-token grids, and computes per-category mean scores.

Output: data/lvis_category_political_scores.csv

Usage:
    python scripts/downstream/extract_lvis_category_scores.py
    python scripts/downstream/extract_lvis_category_scores.py --model qwen3_vl --probe headwise_linear
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_args():
    p = argparse.ArgumentParser(description="Extract per-category LVIS political scores")
    p.add_argument("--model", default="qwen3_vl", choices=["qwen3_vl", "gemma4"])
    p.add_argument("--probe", default="headwise_linear",
                   choices=["headwise_linear", "layerwise_linear", "layerwise_rfm"])
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: data/lvis_category_political_scores.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    lvis_json = os.path.join(ROOT_DIR, "datasets", "lvis", "lvis_v1_train.json")
    csv_path = os.path.join(
        ROOT_DIR, "results", "token_scoring", args.model, "lvis",
        f"prompt_token_stats_combined_ideology_{args.probe}.csv",
    )
    pt_path = os.path.join(
        ROOT_DIR, "results", "token_scoring", args.model, "lvis",
        f"prompt_image_token_scores_combined_ideology_{args.probe}.pt",
    )
    output_path = args.output or os.path.join(ROOT_DIR, "data", "lvis_category_political_scores.csv")

    print(f"Model: {args.model}  |  Probe: {args.probe}")
    print(f"LVIS JSON: {lvis_json}")
    print(f"CSV:       {csv_path}")
    print(f"PT scores: {pt_path}")
    print(f"Output:    {output_path}")

    for p in [lvis_json, csv_path, pt_path]:
        if not os.path.exists(p):
            sys.exit(f"File not found: {p}")

    # --- Load LVIS annotations ---
    print("\nLoading LVIS annotations...")
    with open(lvis_json) as f:
        lvis = json.load(f)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in lvis["categories"]}
    cat_id_to_freq = {cat["id"]: cat["frequency"] for cat in lvis["categories"]}
    print(f"  LVIS categories: {len(cat_id_to_name)}")

    lvis_img_to_anns = defaultdict(list)
    for ann in lvis["annotations"]:
        lvis_img_to_anns[ann["image_id"]].append({
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
        })
    print(f"  LVIS images with annotations: {len(lvis_img_to_anns)}")
    print(f"  Total annotations: {sum(len(v) for v in lvis_img_to_anns.values())}")

    # --- Load per-image CSV stats ---
    print("\nLoading CSV stats...")
    stats = pd.read_csv(csv_path)
    stats = stats.copy()

    # Harmonize columns to the ones the notebook expects
    if "record_name" not in stats.columns and "name" in stats.columns:
        stats["record_name"] = stats["name"]

    stats["record_name"] = stats["record_name"].apply(
        lambda x: os.path.basename(x) if isinstance(x, str) else str(x)
    )

    print(f"  CSV rows: {len(stats)}")

    # --- Load per-token PT scores ---
    print("\nLoading PT scores...")
    pt = torch.load(pt_path, map_location="cpu", weights_only=True)
    all_scores = pt["scores"]
    grid_hw = pt["grid_hw"]
    pt_records = [os.path.basename(r) if isinstance(r, str) else str(r) for r in pt["record_names"]]

    tokens_per_img = grid_hw[:, 0] * grid_hw[:, 1]
    offsets_pt = torch.cat([torch.zeros(1, dtype=torch.long), tokens_per_img.cumsum(0)])

    record_to_idx = {pt_records[i]: i for i in range(len(pt_records))}
    print(f"  Total records: {len(pt_records)}")
    print(f"  Total image tokens: {len(all_scores)}")

    # Build image metadata lookup
    img_meta = {}
    for _, row in stats.iterrows():
        rn = row["record_name"]
        if isinstance(rn, Path):
            rn = rn.name
        img_meta[rn] = {
            "grid_h": int(row["grid_h"]),
            "grid_w": int(row["grid_w"]),
            "image_h": int(row["image_h"]),
            "image_w": int(row["image_w"]),
        }

    # Build image_id -> record_name mapping
    def record_name_to_image_id(rn: str) -> int:
        return int(Path(rn).stem)

    image_id_to_records = defaultdict(list)
    for rn in stats["record_name"]:
        if isinstance(rn, Path):
            rn = rn.name
        img_id = record_name_to_image_id(rn)
        image_id_to_records[img_id].append(rn)

    # --- Map bboxes to token grid ---
    print("\nMapping bboxes to token grid...")
    category_token_scores = defaultdict(list)

    n_matched = 0
    n_unmatched = 0

    for lvis_img_id, anns in lvis_img_to_anns.items():
        if lvis_img_id not in image_id_to_records:
            n_unmatched += 1
            continue
        record_name = image_id_to_records[lvis_img_id][0]
        if record_name not in record_to_idx or record_name not in img_meta:
            n_unmatched += 1
            continue

        meta = img_meta[record_name]
        grid_h, grid_w = meta["grid_h"], meta["grid_w"]
        img_w, img_h = meta["image_w"], meta["image_h"]
        if grid_h * grid_w == 0:
            continue

        idx = record_to_idx[record_name]
        start, end = int(offsets_pt[idx]), int(offsets_pt[idx + 1])
        if end - start != grid_h * grid_w:
            continue

        token_grid = all_scores[start:end].view(grid_h, grid_w).float()

        img_cat_patches = defaultdict(list)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]

            tx0 = int(x / img_w * grid_w)
            ty0 = int(y / img_h * grid_h)
            tx1 = int((x + w) / img_w * grid_w)
            ty1 = int((y + h) / img_h * grid_h)

            tx0 = max(0, min(tx0, grid_w - 1))
            tx1 = max(tx0 + 1, min(tx1, grid_w))
            ty0 = max(0, min(ty0, grid_h - 1))
            ty1 = max(ty0 + 1, min(ty1, grid_h))

            patch = token_grid[ty0:ty1, tx0:tx1]
            if patch.numel() > 0:
                img_cat_patches[cat_id].append(patch)

        for cat_id, patches in img_cat_patches.items():
            all_vals = torch.cat([p.flatten() for p in patches])
            category_token_scores[cat_id].append(all_vals.mean().item())

        n_matched += 1
        if n_matched % 10000 == 0:
            print(f"  Processed {n_matched:,} images...")

    print(f"\n  Images processed: {n_matched:,}")
    print(f"  Images unmatched: {n_unmatched:,}")
    print(f"  Categories with scores: {len(category_token_scores)}")

    # --- Build summary DataFrame ---
    print("\nBuilding summary DataFrame...")
    cat_summary = []
    for cat_id, scores in category_token_scores.items():
        s = np.array(scores)
        cat_summary.append({
            "category_id": cat_id,
            "category_name": cat_id_to_name.get(cat_id, f"unknown_{cat_id}"),
            "frequency": cat_id_to_freq.get(cat_id, "?"),
            "n_images": len(scores),
            "mean_score": s.mean(),
            "median_score": np.median(s),
            "std_score": s.std(),
        })

    df = pd.DataFrame(cat_summary).sort_values("mean_score")
    df = df[df["n_images"] >= 10].copy()
    print(f"  After filtering (>=10 images): {len(df)} categories")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} categories to: {output_path}")

    # Print top/bottom
    print("\n--- Top 10 Republican (highest mean_score) ---")
    for _, row in df.tail(10).iloc[::-1].iterrows():
        print(f"  {row['category_name']:<40s}  mean={row['mean_score']:.4f}  n={int(row['n_images'])}")

    print("\n--- Top 10 Democrat (lowest mean_score) ---")
    for _, row in df.head(10).iterrows():
        print(f"  {row['category_name']:<40s}  mean={row['mean_score']:.4f}  n={int(row['n_images'])}")


if __name__ == "__main__":
    main()
