#!/usr/bin/env python3
"""
Run SAM3 Promptable Concept Segmentation (PCS) on extracted visual features.

Reads a features JSONL file (output of extract_features.py) and for each record
with extracted features, runs SAM3 with each feature as a text prompt to get
segmentation masks. Saves masks to a .npz file and metadata to JSONL.

Usage:
    python scripts/probes/sam3_feature_masks.py --input results/prompt_generation/gemma4/easyportrait_1000samples_max512tok_features.jsonl

    python scripts/probes/sam3_feature_masks.py --input results/prompt_generation/qwen3_vl/easyportrait_1000samples_max512tok_features.jsonl
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SAM3_MODEL_DIR = "/project/jevans/tzhang3/models/sam3"

DEFAULT_PROBE = "combined_ideology_headwise_linear"

MASK_COLORS = [
    (220, 50, 50),
    (50, 180, 50),
    (50, 50, 220),
    (50, 200, 200),
    (220, 220, 50),
    (220, 50, 180),
    (180, 100, 50),
    (100, 50, 180),
]


def build_image_lookup(model: str) -> Dict[str, str]:
    """
    Build a record_id -> image_path lookup from the token scoring CSV.
    Uses the resized images (800px max) to avoid SAM3 shape errors.
    """
    import pandas as pd
    csv_path = os.path.join(
        ROOT_DIR, "results", "token_scoring", model, "easyportrait",
        f"prompt_token_fg_bg_stats_{DEFAULT_PROBE}.csv",
    )
    if not os.path.exists(csv_path):
        print(f"  Warning: CSV not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path, usecols=["record_id", "image_path"])
    lookup = {}
    for _, row in df.iterrows():
        rid = str(row["record_id"])
        lookup[rid] = os.path.join(ROOT_DIR, str(row["image_path"]))
    print(f"  Built image lookup: {len(lookup)} entries from {csv_path}")
    return lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 PCS on extracted visual features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to features JSONL file (output of extract_features.py).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path (default: <input_stem>_sam3_masks.jsonl).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="SAM3 confidence threshold for filtering masks.",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=0.5,
        help="SAM3 threshold for mask binarization.",
    )
    parser.add_argument(
        "--max-features", type=int, default=5,
        help="Max number of features to segment per image.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of records to process (0 = all).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (auto-detected: cuda if available).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing output file.",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--max-image-size", type=int, default=640,
        help="Resize images so the longest edge is at most this many pixels.",
    )
    parser.add_argument(
        "--save-vis", action="store_true", default=False,
        help="Also save masked visualization images (overlay + label per feature).",
    )
    parser.add_argument(
        "--vis-dir", default=None,
        help="Directory for visualization outputs (default: alongside masks npz).",
    )
    return parser.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


from scipy.ndimage import binary_dilation


def _boundary_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Extract boundary (outline) from a binary mask by dilating and subtracting."""
    dilated = binary_dilation(mask, iterations=thickness)
    return (dilated.astype(np.uint8) - mask.astype(np.uint8)).clip(0, 1)


def _overlay_all_masks(pil_image: Image.Image, mask_infos: list) -> Image.Image:
    """Overlay multiple mask boundaries with distinct colors and a legend."""
    image = pil_image.convert("RGBA")
    for mask, color, _label in mask_infos:
        boundary = _boundary_mask(mask)
        boundary_img = Image.fromarray((boundary * 255).astype(np.uint8))
        boundary_img = boundary_img.resize(image.size, Image.NEAREST)
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(boundary_img)
        image = Image.alpha_composite(image, overlay)
    result = image.convert("RGB")

    # Draw legend
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    y = 6
    for _mask, color, label in mask_infos:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 5
        draw.rectangle([4, y, 4 + tw + pad * 2 + 20, y + th + pad], fill=(0, 0, 0, 180))
        draw.rectangle([8, y + 3, 20, y + th + pad - 3], fill=color)
        draw.text((26, y + pad // 2), label, fill=(255, 255, 255), font=font)
        y += th + pad + 4
    return result


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(ROOT_DIR, input_path)

    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_sam3.jsonl"
    masks_npz_path = os.path.splitext(output_path)[0] + "_masks.npz"

    if os.path.exists(output_path) and not args.overwrite:
        print(f"Output exists: {output_path}  (use --overwrite to re-run)")
        return

    print(f"Loading records from {input_path} ...")
    records = load_records(input_path)
    n_total = len(records)

    records_with_features = [
        (i, r) for i, r in enumerate(records)
        if r.get("extracted_features") and r.get("extracted_features") != ["None"]
    ]
    if args.limit > 0:
        records_with_features = records_with_features[:args.limit]
        print(f"  Limited to {args.limit} records")
    # Determine model name from input path (e.g. qwen3_vl or gemma4)
    input_rel = os.path.relpath(input_path, os.path.join(ROOT_DIR, "results", "prompt_generation"))
    model = input_rel.split(os.sep)[0] if os.sep in input_rel else "qwen3_vl"
    print(f"Detected model: {model}")

    # Build image lookup from token scoring CSV (resized 800px images)
    image_lookup = build_image_lookup(model)

    # Fetch resized image dir directly for model
    resized_dir = os.path.join(
        ROOT_DIR, "results", "token_scoring", model, "easyportrait", "_resized_images_800",
    )
    has_resized = os.path.isdir(resized_dir)

    print(f"Loaded {n_total} records ({len(records_with_features)} with features)")

    # Load SAM3
    from transformers import Sam3Model, Sam3Processor

    print(f"Loading SAM3 from {SAM3_MODEL_DIR} ({args.dtype})...")
    model = Sam3Model.from_pretrained(SAM3_MODEL_DIR, torch_dtype=dtype).to(device)
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_DIR)
    model.eval()
    print("Model loaded.")

    success = 0
    skipped_image = 0
    skipped_no_mask = 0
    errors = 0
    total_start = time.time()

    results: List[Dict[str, Any]] = []
    all_masks: Dict[str, np.ndarray] = {}

    # Prepare vis output directory if saving visualizations
    vis_dir = None
    if args.save_vis:
        vis_dir = args.vis_dir or os.path.splitext(output_path)[0] + "_vis"
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  Saving visualizations to: {vis_dir}")

    pbar = tqdm(records_with_features, desc="Segmenting features")
    for orig_idx, rec in pbar:
        record_id = rec.get("record_id", f"record_{orig_idx}")
        features = rec["extracted_features"]
        if isinstance(features, str):
            features = [features]

        # Truncate to max features
        features = features[: args.max_features]
        feature_masks: Dict[str, Dict] = {}

        # Resolve image path: use CSV lookup first, then fall back to resized dir
        image_path = image_lookup.get(record_id)
        if not image_path and has_resized:
            image_name = rec.get("name", "")
            if image_name:
                image_path = os.path.join(resized_dir, image_name)
        if not image_path or not os.path.exists(image_path):
            skipped_image += 1
            results.append({
                **rec,
                "_sam3_error": "image not found",
                "feature_masks": {},
            })
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            skipped_image += 1
            print(f"\n  [ERROR] record={record_id} failed to load image {image_path}: {exc}", file=sys.stderr)
            results.append({
                **rec,
                "_sam3_error": "failed to load image",
                "feature_masks": {},
            })
            continue

        # Resize if image is larger than max_image_size
        max_dim = max(image.size)
        if max_dim > args.max_image_size:
            scale = args.max_image_size / max_dim
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.LANCZOS)

        orig_size = image.size  # (W, H)

        # Run SAM3 PCS one image x feature at a time
        mask_infos = []
        for fi, fname in enumerate(features):
            try:
                processor_inputs = processor(
                    images=image,
                    text=fname,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    sam3_outputs = model(**processor_inputs)

                img_results = processor.post_process_instance_segmentation(
                    sam3_outputs,
                    threshold=args.threshold,
                    mask_threshold=args.mask_threshold,
                    target_sizes=processor_inputs.get("original_sizes").tolist(),
                )
            except Exception as exc:
                errors += 1
                print(f"\n  [ERROR] record={record_id} feature={fname}: {exc}", file=sys.stderr)
                continue

            img_result = img_results[0]
            masks = img_result.get("masks", [])
            scores = img_result.get("scores", [])

            if len(masks) == 0:
                continue

            # Store top mask per feature
            best_idx = 0
            if len(scores) > 0:
                best_idx = int(torch.argmax(scores).item()) if isinstance(scores, torch.Tensor) else int(np.argmax(scores))

            mask = masks[best_idx]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

            if mask.sum() == 0:
                continue

            mask_key = f"{record_id}/{fname}"
            all_masks[mask_key] = mask
            feature_masks[fname] = {"shape": list(mask.shape)}

            if vis_dir:
                color = MASK_COLORS[fi % len(MASK_COLORS)]
                mask_infos.append((mask, color, fname))

        if vis_dir and mask_infos:
            vis_img = _overlay_all_masks(image, mask_infos)
            vis_img.save(os.path.join(vis_dir, f"{record_id}.png"))

        if feature_masks:
            success += 1
        else:
            skipped_no_mask += 1

        results.append({**rec, "feature_masks": feature_masks})
        pbar.set_postfix(success=success, no_mask=skipped_no_mask, err=errors)

    total_elapsed = time.time() - total_start

    # Save masks to npz
    os.makedirs(os.path.dirname(masks_npz_path) or ".", exist_ok=True)
    np.savez_compressed(masks_npz_path, **all_masks)

    # Save metadata JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n--- Summary ---")
    print(f"  Total:          {n_total}")
    print(f"  With masks:     {success}")
    print(f"  No masks found: {skipped_no_mask}")
    print(f"  Image issues:   {skipped_image}")
    print(f"  Errors:         {errors}")
    print(f"  Wall time:      {total_elapsed:.1f}s")
    print(f"  Masks npz:      {masks_npz_path}")
    print(f"  Metadata JSONL: {output_path}")


if __name__ == "__main__":
    main()
