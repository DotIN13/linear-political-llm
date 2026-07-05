#!/usr/bin/env python3
"""
Image Token Pruning Experiment (Gemma4).

Renders text from twinviews-13k.csv onto canvas images, then measures
how removing image soft tokens (by zeroing or dropping) degrades the
model's ability to transcribe the text.

Scenarios (7 total):
  0: baseline        – no modification
  1: zero_25         – zero out 25% of image features
  2: zero_50         – zero out 50% of image features
  3: zero_75         – zero out 75% of image features
  4: drop_25         – drop    25% of image tokens (shorter sequence)
  5: drop_50         – drop    50% of image tokens
  6: drop_75         – drop    75% of image tokens

Usage:
    python scripts/probes/prune_image_tokens.py --n-samples 10

Outputs:
    results/prune_image_tokens/scores.csv
    results/prune_image_tokens/viz_sample_{i:02d}.png
    results/prune_image_tokens/summary_chart.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants ───────────────────────────────────────────────────────────
MODEL_PATH = "/project/jevans/tzhang3/models/gemma-4-31B-it"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DATA_PATH = ROOT / "data" / "twinviews-13k.csv"
OUTPUT_DIR = ROOT / "results" / "prune_image_tokens"

IMAGE_TOKEN_ID = 258880        # <|image|> placeholder
IMAGE_MARKER_ID = 255999       # <|image> start marker

CANVAS_MAX_WIDTH = 900
FONT_SIZE = 32
LINE_SPACING = 4
MAX_NEW_TOKENS = 64
PROMPT = "Transcribe the text in this image exactly:"

RATIOS = [0.25, 0.50, 0.75]
DEFAULT_STRATEGIES = ["random", "block"]


def build_scenarios(strategies: List[str]) -> List[Tuple[str, str, float, Optional[str]]]:
    """Build scenario list from strategy names."""
    specs = [("baseline", "baseline", 0.00, None)]
    for strat in strategies:
        for r in RATIOS:
            specs.append((f"{strat}_zero_{int(r*100):02d}", "zero", r, strat))
            specs.append((f"{strat}_drop_{int(r*100):02d}", "drop", r, strat))
    return specs


SCENARIOS = build_scenarios(DEFAULT_STRATEGIES)

# ── Helpers ──────────────────────────────────────────────────────────────
def load_font() -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except (OSError, IOError):
        return ImageFont.load_default()


def word_wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = cur + (" " if cur else "") + w
        if font.getbbox(test)[2] - font.getbbox(test)[0] > max_width - 20:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


def create_text_canvas(
    text: str,
    max_width: int = CANVAS_MAX_WIDTH,
    font_size: int = FONT_SIZE,
) -> Image.Image:
    font = load_font()
    lines = word_wrap(text, font, max_width)
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + LINE_SPACING
    canvas_h = max(len(lines) * line_h + 20, 60)
    canvas = Image.new("RGB", (max_width, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = 10
    for line in lines:
        draw.text((10, y), line, fill="black", font=font)
        y += line_h
    return canvas


def load_model_and_processor():
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    model._orig_get_image_features = model.model.get_image_features
    return model, proc


def encode_image(image: Image.Image, prompt: str, proc) -> dict:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    return proc.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )


def get_image_grid_shape(encoded: dict, proc) -> Tuple[int, int]:
    """
    Returns (grid_h, grid_w) of the pooled image features.
    For Gemma4, derived from image_position_ids.
    """
    if "image_position_ids" in encoded:
        ipos = encoded["image_position_ids"][0]  # [N, 2]
        valid = (ipos[:, 0] >= 0) & (ipos[:, 1] >= 0)
        max_x = int(ipos[valid, 0].max().item() + 1)
        max_y = int(ipos[valid, 1].max().item() + 1)
        pool_k = getattr(proc.image_processor, "pooling_kernel_size", 1)
        return int(max_y // int(pool_k)), int(max_x // int(pool_k))
    raise ValueError("No image_position_ids in encoded dict")


def build_keep_mask(n_tokens: int, ratio: float, device,
                    grid_h: int = None, grid_w: int = None,
                    strategy: str = "random") -> torch.Tensor:
    """
    Keep mask (True = keep).
    strategy='random': randomly drop individual tokens.
    strategy='block':   drop one contiguous rectangular region.
    strategy='row':     drop a contiguous horizontal strip.
    """
    if ratio == 0.0:
        return torch.ones(n_tokens, dtype=torch.bool, device=device)

    if strategy in ("block", "row") and grid_h is not None and grid_w is not None:
        return _build_area_mask(grid_h, grid_w, ratio, device, strategy)

    # Random independent fallback
    n_keep = max(int(n_tokens * (1.0 - ratio)), 1)
    perm = torch.randperm(n_tokens, device=device)
    keep = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    keep[perm[:n_keep]] = True
    return keep


def _build_area_mask(grid_h: int, grid_w: int, ratio: float, device,
                     strategy: str) -> torch.Tensor:
    """Build a 2D block/row mask, then flatten row-major to 1D."""
    total = grid_h * grid_w
    n_remove = int(total * ratio)
    if n_remove < 1:
        n_remove = 1
    keep_2d = torch.ones(grid_h, grid_w, dtype=torch.bool, device=device)

    if strategy == "block":
        # Choose a random rectangle covering roughly n_remove cells
        # Pick random height between 1 and grid_h, compute width ≈ n_remove / height
        h = torch.randint(1, min(grid_h, n_remove) + 1, (1,), device=device).item()
        h = min(h, grid_h)
        w = min(max(int(n_remove / h), 1), grid_w)
        # Clamp so product doesn't exceed n_remove too much (allow overshoot)
        if h * w > total:
            h, w = grid_h // 2, grid_w // 2

        r0 = torch.randint(0, grid_h - h + 1, (1,), device=device).item() if grid_h > h else 0
        c0 = torch.randint(0, grid_w - w + 1, (1,), device=device).item() if grid_w > w else 0
        keep_2d[r0:r0 + h, c0:c0 + w] = False

    elif strategy == "row":
        # Drop a contiguous horizontal strip of n_rows
        n_rows = max(1, int(grid_h * ratio))
        n_rows = min(n_rows, grid_h)
        r0 = torch.randint(0, grid_h - n_rows + 1, (1,), device=device).item() if grid_h > n_rows else 0
        keep_2d[r0:r0 + n_rows, :] = False

    # Flatten row-major
    return keep_2d.reshape(-1)


def compute_metrics(ground_truth: str, predicted: str) -> dict:
    """CER (character error rate), WER (word error rate), exact match."""
    if not predicted:
        return {"cer": 1.0, "wer": 1.0, "exact": False, "n_char_gt": len(ground_truth)}

    # CER
    gt_chars = ground_truth.lower()
    pred_chars = predicted.lower()
    dist = _levenshtein(gt_chars, pred_chars)
    cer = dist / max(len(gt_chars), 1)

    # WER
    gt_words = ground_truth.lower().split()
    pred_words = predicted.lower().split()
    word_dist = _levenshtein(gt_words, pred_words)
    wer = word_dist / max(len(gt_words), 1)

    exact = (ground_truth.strip().lower() == predicted.strip().lower())

    return {"cer": cer, "wer": wer, "exact": exact, "n_char_gt": len(gt_chars)}


def _levenshtein(a, b) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ── Main experiment logic ────────────────────────────────────────────────

def run_scenario(
    model, proc, encoded, enc_dev, orig_get_image_features,
    n_img: int, device, scenario_name: str, method: str, ratio: float,
    keep_mask: Optional[torch.Tensor] = None,
) -> Tuple[str, torch.Tensor]:
    """
    Run one pruning scenario. Returns (generated_text, keep_mask_used).

    For 'baseline': no modification.
    For 'zero':    monkey-patch get_image_features to multiply by keep_mask.
    For 'drop':    monkey-patch get_image_features to subset by keep_mask,
                   and shorten input_ids/mm_token_type_ids.
    """
    if method == "baseline":
        model.model.get_image_features = lambda pv, pid, **kw: orig_get_image_features(pv, pid, **kw)
        with torch.no_grad():
            out = model.generate(**enc_dev, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        gen_text = proc.tokenizer.decode(out[0, enc_dev["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        keep_mask_vis = torch.ones(n_img, dtype=torch.bool, device=device)
        return gen_text, keep_mask_vis

    if method == "zero":
        mask = keep_mask.to(device)
        def zero_patch(pv, pid, **kw):
            out = orig_get_image_features(pv, pid, **kw)
            out.pooler_output = out.pooler_output * mask.unsqueeze(-1).to(out.pooler_output.dtype)
            return out
        model.model.get_image_features = zero_patch
        with torch.no_grad():
            out = model.generate(**enc_dev, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        gen_text = proc.tokenizer.decode(out[0, enc_dev["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return gen_text, mask

    if method == "drop":
        mask = keep_mask.to(device)
        n_kept = mask.sum().item()

        # Build shorter input_ids
        ids = enc_dev["input_ids"][0]
        is_img_mask = ids == IMAGE_TOKEN_ID
        img_positions = torch.where(is_img_mask)[0]
        img_start = img_positions[0].item()
        img_end = img_positions[-1].item() + 1

        before_ids = ids[:img_start].cpu()
        after_ids = ids[img_end:].cpu()
        img_block = torch.full((n_kept,), IMAGE_TOKEN_ID, dtype=torch.long)
        new_ids = torch.cat([before_ids, img_block, after_ids]).unsqueeze(0).to(device)

        new_mm = torch.zeros(1, new_ids.shape[1], dtype=torch.long, device=device)
        new_mm[:, before_ids.shape[0]:before_ids.shape[0] + n_kept] = 1
        new_attn = torch.ones(1, new_ids.shape[1], dtype=torch.long, device=device)

        def drop_patch(pv, pid, **kw):
            out = orig_get_image_features(pv, pid, **kw)
            out.pooler_output = out.pooler_output[mask]
            return out
        model.model.get_image_features = drop_patch

        drop_inputs = {
            "input_ids": new_ids,
            "pixel_values": enc_dev["pixel_values"],
            "image_position_ids": enc_dev["image_position_ids"],
            "mm_token_type_ids": new_mm,
            "attention_mask": new_attn,
        }
        with torch.no_grad():
            out = model.generate(**drop_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        gen_text = proc.tokenizer.decode(out[0, new_ids.shape[1]:], skip_special_tokens=True).strip()
        return gen_text, mask


def process_sample(
    text: str, idx: int, model, proc, device, scenarios: List[Tuple[str, str, float, Optional[str]]],
) -> dict:
    """Run all scenarios on one text sample."""
    # Restore original method (may have been patched by previous sample)
    model.model.get_image_features = model._orig_get_image_features

    canvas = create_text_canvas(text)
    encoded = encode_image(canvas, PROMPT, proc)
    enc_dev = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}

    orig_get_image_features = model._orig_get_image_features

    with torch.no_grad():
        ref_feats = orig_get_image_features(enc_dev["pixel_values"], enc_dev["image_position_ids"])
        n_img = ref_feats.pooler_output.shape[0]

    grid_h, grid_w = get_image_grid_shape(encoded, proc)

    # Precompute keep-masks per (strategy, ratio) so zero/drop use identical positions
    masks_cache = {}
    for _, method, ratio, strategy in scenarios:
        if method == "baseline":
            continue
        key = (strategy, ratio, grid_h, grid_w)
        if key not in masks_cache:
            masks_cache[key] = build_keep_mask(n_img, ratio, device,
                                               grid_h=grid_h, grid_w=grid_w,
                                               strategy=strategy or "random")

    results = []
    for scenario_name, method, ratio, strategy in scenarios:
        if method == "baseline":
            keep_mask = None
        else:
            keep_mask = masks_cache[(strategy, ratio, grid_h, grid_w)]
        gen_text, mask_used = run_scenario(
            model, proc, encoded, enc_dev, orig_get_image_features,
            n_img, device, scenario_name, method, ratio, keep_mask,
        )
        metrics = compute_metrics(text, gen_text)
        results.append({
            "sample_idx": idx,
            "scenario": scenario_name,
            "method": method,
            "ratio": ratio,
            "strategy": strategy or "—",
            "ground_truth": text,
            "predicted": gen_text,
            **metrics,
            "n_img_tokens": n_img,
            "n_img_kept": mask_used.sum().item() if mask_used is not None else n_img,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "canvas_w": canvas.width,
            "canvas_h": canvas.height,
        })
    return {"results": results, "canvas": canvas,
            "masks_cache": masks_cache,
            "grid_h": grid_h, "grid_w": grid_w,
            "pool_k": getattr(proc.image_processor, "pooling_kernel_size", 1)}


# ── Visualization ────────────────────────────────────────────────────────

def make_token_overlay(canvas: Image.Image, mask_1d: torch.Tensor,
                       grid_h: int, grid_w: int, alpha: float = 0.55) -> Image.Image:
    """
    Overlay a heatmap of kept/removed tokens on the canvas.
    mask_1d[N]: True = kept, False = removed. Assumes row-major ordering.
    """
    if mask_1d.sum() == len(mask_1d):
        return canvas  # no removal

    mask_2d = mask_1d.reshape(grid_h, grid_w).float().cpu().numpy()
    canvas_rgb = canvas.convert("RGB")
    img = np.asarray(canvas_rgb).astype(np.float32) / 255.0

    # Upsample mask to image size
    mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8))
    mask_up = np.array(mask_img.resize(
        (canvas_rgb.width, canvas_rgb.height), Image.NEAREST
    )).astype(np.float32) / 255.0

    # Red = removed, green = kept
    overlay = np.zeros((*img.shape[:2], 3), dtype=np.float32)
    overlay[..., 0] = (1.0 - mask_up) * 0.9   # red where removed
    overlay[..., 1] = mask_up * 0.1            # green where kept
    overlay_weight = np.abs(mask_up - 0.5) * 2.0 * alpha  # stronger at extremes

    result = img * (1 - overlay_weight[..., None]) + overlay * overlay_weight[..., None]
    result = np.clip(result, 0, 1)
    return Image.fromarray((result * 255).astype(np.uint8))


def visualize_sample(sample_data: dict, output_path: str):
    """Create a multi-panel figure for one text sample with mask overlays."""
    canvas = sample_data["canvas"]
    results = pd.DataFrame(sample_data["results"])
    masks_cache = sample_data.get("masks_cache", {})
    grid_h = sample_data["grid_h"]
    grid_w = sample_data["grid_w"]

    n_scenarios = len(results)
    fig, axes = plt.subplots(n_scenarios + 1, 4, figsize=(14, 2 + 2.5 * (n_scenarios + 1)),
                              gridspec_kw={"width_ratios": [3.5, 3.5, 1.5, 1.5],
                                           "hspace": 0.3, "wspace": 0.25})
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

    # Row 0: original canvas + summary
    gt_text = results.iloc[0]["ground_truth"]
    _set_row(axes[0], "Original", canvas, None, gt_text, np.nan, np.nan, True)

    # Data rows
    for i, (_, row) in enumerate(results.iterrows()):
        label = row["scenario"]
        method = row["method"]
        ratio = row["ratio"]
        strategy = row["strategy"]
        n_img = int(row["n_img_tokens"])

        # Build overlay image from cached mask
        overlay_img = None
        if method != "baseline":
            key = (strategy, ratio, grid_h, grid_w)
            mask = masks_cache.get(key)
            if mask is not None and mask.sum() < len(mask):
                overlay_img = make_token_overlay(canvas, mask, grid_h, grid_w)

        _set_row(axes[i + 1], label, canvas, overlay_img, row["predicted"],
                 row["cer"], row["wer"], False)

    plt.suptitle(f"Token Pruning Results — Ground truth:\n\"{gt_text}\"",
                 fontsize=11, fontweight="bold", y=1.01)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _set_row(axs, label, canvas, overlay, text, cer, wer, is_header):
    """Fill one row of subplots."""
    # Column 0: scenario name
    axs[0].axis("off")
    axs[0].text(0.5, 0.5, label, ha="center", va="center",
                fontsize=11, fontweight="bold")

    # Column 1: canvas image (with optional overlay)
    img = np.asarray(overlay.convert("RGB") if overlay else canvas)
    axs[1].imshow(img)
    axs[1].axis("off")

    # Column 2: predicted text
    axs[2].axis("off")
    axs[2].text(0.05, 0.5, text if text else "(empty)", ha="left", va="center",
                wrap=True, fontsize=8,
                fontstyle="italic" if not text else "normal",
                color="gray" if not text else "black")

    # Column 3: metrics
    axs[3].axis("off")
    if is_header:
        return
    cer_str = f"CER: {cer:.3f}" if not np.isnan(cer) else "CER: —"
    wer_str = f"WER: {wer:.3f}" if not np.isnan(wer) else "WER: —"
    axs[3].text(0.5, 0.65, cer_str, ha="center", va="center", fontsize=9)
    axs[3].text(0.5, 0.35, wer_str, ha="center", va="center", fontsize=9)


def visualize_summary(all_results: pd.DataFrame, output_dir: str):
    """Plot CER/WER vs ratio, faceted by strategy and method (zero/drop)."""
    df = all_results[all_results["method"] != "baseline"].copy()
    strategies_in_use = [s for s in df["strategy"].unique() if s != "—"]
    if not strategies_in_use:
        return

    n_strat = len(strategies_in_use)
    fig, axes = plt.subplots(n_strat, 2, figsize=(10, 3 + 3 * n_strat),
                              squeeze=False)
    plt.rcParams.update({"font.size": 9})

    colors = {"zero": "#4E79A7", "drop": "#E15759"}
    markers = {"zero": "o", "drop": "s"}

    for row_idx, strat in enumerate(strategies_in_use):
        subdf = df[df["strategy"] == strat]
        for col_idx, metric in enumerate(["cer", "wer"]):
            ax = axes[row_idx][col_idx]
            for method in ["zero", "drop"]:
                sub = subdf[subdf["method"] == method]
                grouped = sub.groupby("ratio")[metric].agg(["mean", "std"]).reset_index()
                ax.errorbar(
                    grouped["ratio"], grouped["mean"], yerr=grouped["std"],
                    fmt=markers[method] + "-", color=colors[method],
                    capsize=4, markersize=6, linewidth=1.3, label=method,
                )
            ax.set_xlabel("Pruning ratio")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{strat} — {metric.upper()}")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(bottom=-0.02)

    plt.suptitle("Image Token Pruning — Transcription Degradation", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_chart.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Summary chart saved: {path}")


# ── Logging ──────────────────────────────────────────────────────────────

def write_sample_log(sample_data: dict, idx: int, log_path: str, file_mode: str = "a"):
    results = sample_data["results"]
    canvas = sample_data["canvas"]
    pool_k = sample_data.get("pool_k", 3)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, file_mode, encoding="utf-8") as f:
        f.write(f"{'─' * 70}\n")
        f.write(f"Sample {idx:02d}\n")
        f.write(f"{'─' * 70}\n")
        gt = results[0]["ground_truth"]
        f.write(f"Ground truth:  {gt}\n")
        f.write(f"Canvas:        {canvas.width}×{canvas.height} px\n")
        f.write(f"Image tokens:  {results[0]['n_img_tokens']}\n")
        f.write(f"Grid:          {results[0]['grid_h']}×{results[0]['grid_w']} "
                f"(pooled, kernel={pool_k})\n")
        f.write(f"\n{'Scenario':<22s} {'Method':<6s} {'Ratio':>5s}  {'CER':>6s} {'WER':>6s} {'Exact':>6s}  Generation\n")
        f.write(f"{'─' * 22} {'─' * 6} {'─' * 5}  {'─' * 6} {'─' * 6} {'─' * 6}  {'─' * 40}\n")
        for r in results:
            cer = f"{r['cer']:.3f}" if r['cer'] is not None else "—"
            wer = f"{r['wer']:.3f}" if r['wer'] is not None else "—"
            exact = "✓" if r['exact'] else "✗"
            pred = r["predicted"] if r["predicted"] else "(empty)"
            f.write(f"{r['scenario']:<22s} {r['method']:<6s} {r['ratio']:>5.2f}  {cer:>6s} {wer:>6s} {exact:>6s}  {pred}\n")
        f.write("\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Image Token Pruning Experiment")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of text samples from twinviews CSV")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results and visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-viz", action="store_true", help="Skip per-sample visualizations")
    parser.add_argument("--strategies", type=str, nargs="+", default=DEFAULT_STRATEGIES,
                        choices=["random", "block", "row"],
                        help="Token removal strategies (default: random block)")
    args = parser.parse_args()

    scenarios = build_scenarios(args.strategies)
    print(f"Strategies: {args.strategies}  →  {len(scenarios)} scenarios per sample")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Sample texts (use column 'l' — left-leaning perspectives)
    texts = df["l"].dropna().tolist()
    texts = [t for t in texts if 30 < len(t) < 400]
    sampled = np.random.RandomState(args.seed).choice(texts, size=min(args.n_samples, len(texts)), replace=False)
    sampled = list(sampled)
    print(f"Sampled {len(sampled)} texts (filtered to {len(texts)} candidates)")

    # Load model
    print("Loading Gemma4 model...")
    model, proc = load_model_and_processor()
    device = next(model.parameters()).device
    print(f"  Model on {device}")

    # Run experiments
    all_records = []
    log_path = os.path.join(args.output_dir, "generations.log")
    _log_mode = "w"
    for i, text in enumerate(tqdm(sampled, desc="Processing samples")):
        print(f"\n--- Sample {i:02d} ---")
        print(f"  Text: {text[:80]}...")
        sample_data = process_sample(text, i, model, proc, device, scenarios)
        all_records.extend(sample_data["results"])

        write_sample_log(sample_data, i, log_path, file_mode=_log_mode)
        _log_mode = "a"

        if not args.skip_viz:
            viz_path = os.path.join(args.output_dir, f"viz_sample_{i:02d}.png")
            visualize_sample(sample_data, viz_path)

    # Save scores
    scores_df = pd.DataFrame(all_records)
    scores_path = os.path.join(args.output_dir, "scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"\nScores saved: {scores_path}")
    print(f"Log saved:    {log_path}")

    # Print summary
    print("\n=== Summary ===")
    summary = scores_df.groupby(["scenario"])[["cer", "wer", "exact"]].mean().round(3)
    summary["n"] = scores_df.groupby("scenario").size()
    print(summary.to_string())

    # Visualize summary
    visualize_summary(scores_df, args.output_dir)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
