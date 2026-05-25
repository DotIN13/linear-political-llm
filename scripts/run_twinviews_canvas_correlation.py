#!/usr/bin/env python3
"""
Multi-Sample: Image Canvas vs Direct Text Correlation.

Runs N twinview pairs through BOTH pipelines:
1. Image canvas: render L/R text as an image, score vision tokens, extract per-sentence region means
2. Direct text: tokenize each sentence separately, compute per-sentence mean

Then computes sentence-level, token-level, and column-level correlations.

Usage:
    python scripts/run_twinviews_canvas_correlation.py --model-families qwen3-vl gemma4 --top-k-values 8 16 32
"""
import argparse
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from probes import HeadwiseLinearProbe
from scripts.probes.token_scoring import (
    build_probe_runtime,
    capture_module_outputs,
    encode_prompts,
    gather_candidate_image_token_ids,
    move_to_device,
    reconstruct_grid_hw,
    select_model_loader,
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
_MODEL_PATHS = {
    "qwen3-vl": "/project/jevans/tzhang3/models/Qwen3-VL-8B-Instruct",
    "gemma4": "/project/jevans/tzhang3/models/gemma-4-31B-it",
}

PROBE_PREFIX = "textual_ideology"
PROBE_TYPE = "headwise_linear"
DATA_DIR = str(ROOT / "results" / "probes")
TV_PATH = ROOT / "data" / "twinviews-13k.csv"

N_ITERS = 50
PAIRS_PER_ITER = 4

MAX_WIDTH = 900
LINE_SPACING = 4

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE = 32

OUTPUT_DIR = ROOT / "results" / "twinviews_canvas_correlation"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_font() -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except (OSError, IOError):
        return ImageFont.load_default()


def build_model(model_family: str):
    model_path = _MODEL_PATHS[model_family]
    model_cls = select_model_loader(model_family, model_path)

    if model_family == "qwen3-vl":
        load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    else:
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    model = model_cls.from_pretrained(model_path, **load_kwargs).eval()
    return model, model_path


def tokenize_text(text, processor, model):
    enc = processor.tokenizer(text, return_tensors="pt")
    return {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in enc.items()
    }


def score_tokens(inputs, model, module_names, runtime):
    captured = capture_module_outputs(
        model=model, encoded=inputs, module_names=module_names,
    )
    from scripts.probes.token_scoring import score_from_captured
    scores = score_from_captured(captured, runtime)
    return scores.cpu().numpy()[0]


def tokenize_and_score(text, processor, model, module_names, runtime):
    inputs = tokenize_text(text, processor, model)
    scores = score_tokens(inputs, model, module_names, runtime)
    ids = inputs["input_ids"][0].cpu().numpy()
    toks = [processor.tokenizer.decode([tid]) for tid in ids]
    if len(scores) > len(toks):
        scores = scores[: len(toks)]
    elif len(scores) < len(toks):
        scores = np.pad(scores, (0, len(toks) - len(scores)), constant_values=np.nan)
    return toks, scores


def score_tokens_with_tokenizer(inputs, model, module_names, runtime):
    captured = capture_module_outputs(
        model=model, encoded=inputs, module_names=module_names,
    )
    from scripts.probes.token_scoring import score_from_captured
    scores = score_from_captured(captured, runtime)
    return scores.cpu().numpy()[0]


def word_wrap(text, clr, font, max_width):
    wrapped_lines = []
    wrapped_colors = []
    for t, c in zip([text], [clr]):
        words = t.split()
        cur = ""
        for w in words:
            test = cur + (" " if cur else "") + w
            if font.getbbox(test)[2] - font.getbbox(test)[0] > max_width - 20:
                if cur:
                    wrapped_lines.append(cur)
                    wrapped_colors.append(c)
                cur = w
            else:
                cur = test
        if cur:
            wrapped_lines.append(cur)
            wrapped_colors.append(c)
    return wrapped_lines, wrapped_colors


def build_canvas(
    sample_pairs, font, line_h, max_width
) -> Tuple[Image.Image, List[str], List[str], List[int], List[str], int]:
    lines = []
    for _, row in sample_pairs.iterrows():
        lines.append((f"L: {row['l']}", "#5bc0de"))
        lines.append((f"R: {row['r']}", "#d9534f"))

    wrapped_lines = []
    wrapped_colors = []
    wrapped_pair_idx = []
    wrapped_side = []
    pi = 0
    for text, clr in lines:
        wl, wc = word_wrap(text, clr, font, max_width)
        wrapped_lines.extend(wl)
        wrapped_colors.extend(wc)
        wrapped_pair_idx.extend([pi] * len(wl))
        side = "L" if "#5bc0de" in clr else "R"
        wrapped_side.extend([side] * len(wl))
        if "#d9534f" in clr:
            pi += 1

    canvas_h = len(wrapped_lines) * line_h + 20
    canvas = Image.new("RGB", (max_width, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = 10
    for text, clr in zip(wrapped_lines, wrapped_colors):
        draw.text((10, y), text, fill=clr, font=font)
        y += line_h

    return canvas, wrapped_lines, wrapped_colors, wrapped_pair_idx, wrapped_side, canvas_h


# ---------------------------------------------------------------------------
# Per-model per-k pipeline
# ---------------------------------------------------------------------------
def run_for_model_k(
    model_family: str,
    top_k: int,
    tv_df: pd.DataFrame,
    font: ImageFont.FreeTypeFont,
    line_h: int,
    output_dir: str,
):
    print(f"\n{'='*60}")
    print(f"Model: {model_family}  |  top_k: {top_k}")
    print(f"{'='*60}")

    from transformers import AutoProcessor

    model, model_path = build_model(model_family)
    processor = AutoProcessor.from_pretrained(model_path)

    # Load probe
    probe = HeadwiseLinearProbe(
        model_path=model_path,
        prefix=PROBE_PREFIX,
        mode="text",
        model_family=model_family,
        data_dir=DATA_DIR,
    ).load()

    runtime = build_probe_runtime(
        model=model, probe=probe, top_k=top_k, mode="text", model_family=model_family,
    )
    module_names = sorted(runtime["module_names"])

    print(f"Probe: {probe.scores_.shape[0]}L x {probe.scores_.shape[1]}H")
    print(f"  Best r: {probe.scores_.max():.4f}  |  top_{top_k} heads across {len(module_names)} layers")

    # Image token IDs
    image_token_ids = gather_candidate_image_token_ids(processor.tokenizer)
    image_token_array = np.asarray(sorted(image_token_ids), dtype=np.int64)

    # Filter short pairs
    sharp = tv_df["l"].apply(len) + tv_df["r"].apply(len)
    tv_filt = tv_df.loc[sharp[sharp < 400].index]

    records = []
    for it in range(N_ITERS):
        sample_pairs = tv_filt.sample(n=PAIRS_PER_ITER, random_state=it).reset_index(drop=True)

        canvas, wrapped_lines, wrapped_colors, wrapped_pair_idx, wrapped_side, canvas_h = build_canvas(
            sample_pairs, font, line_h, MAX_WIDTH
        )

        # Score via VLM
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": canvas},
                    {"type": "text", "text": "What political perspectives are expressed in this image?"},
                ],
            }
        ]
        enc = encode_prompts(processor, [msgs])
        inputs = move_to_device(enc, model)

        input_ids_np = enc["input_ids"][0].cpu().numpy()
        is_image = np.isin(input_ids_np, image_token_array)

        # qwen3-vl needs a real file path to compute grid dimensions
        tmp_path = None
        canvas_path = ""
        try:
            if model_family == "qwen3-vl":
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    canvas.save(tmp.name)
                    canvas_path = tmp.name
                    tmp_path = tmp.name
            gh, gw = reconstruct_grid_hw(model_family, canvas_path, enc, processor)
        finally:
            if tmp_path:
                os.unlink(tmp_path)

        all_scores = score_tokens_with_tokenizer(inputs, model, module_names, runtime)
        img_scores = all_scores[is_image]
        img_grid = img_scores.reshape(gh, gw)

        # Map grid rows to pairs/sentences
        grid_row_height = canvas_h / gh
        row_info = [(-1, "") for _ in range(gh)]
        for li in range(len(wrapped_lines)):
            y_start = 10 + li * line_h
            y_end = y_start + line_h - LINE_SPACING
            row_start = int(y_start / grid_row_height)
            row_end = min(int(y_end / grid_row_height) + 1, gh)
            for r in range(row_start, row_end):
                if 0 <= r < gh:
                    row_info[r] = (wrapped_pair_idx[li], wrapped_side[li])

        # Extract per-sentence data
        for pp, (_, row) in enumerate(sample_pairs.iterrows()):
            for side, key in [("L", "l"), ("R", "r")]:
                _, tscores = tokenize_and_score(row[key], processor, model, module_names, runtime)
                txt_mean = np.nanmean(tscores)

                img_rows = [r for r in range(gh) if row_info[r] == (pp, side)]
                if img_rows:
                    img_region = img_grid[img_rows, :]
                    img_mean = float(img_region.mean())
                    img_flat = img_region.flatten().tolist()
                else:
                    img_mean = float("nan")
                    img_flat = []

                records.append(
                    {
                        "iter": it,
                        "pair_idx": pp,
                        "side": side,
                        "txt_scores": tscores.tolist() if len(tscores) > 0 else [],
                        "txt_mean": txt_mean,
                        "img_n_cells": len(img_flat),
                        "img_mean": img_mean,
                        "img_flat": img_flat,
                    }
                )

        if (it + 1) % 10 == 0:
            print(f"  iter {it + 1}/{N_ITERS}  ({len(records)} sentences so far)")

    df_records = pd.DataFrame(records)
    valid = df_records.dropna(subset=["txt_mean", "img_mean"])
    print(f"Done: {len(df_records)} total, {len(valid)} valid")

    # ---- Sentence-Level Correlation ---- 
    r_all, p_all = stats.pearsonr(valid["txt_mean"], valid["img_mean"])
    rho_all, p_rho_all = stats.spearmanr(valid["txt_mean"], valid["img_mean"])

    r_l, p_l = stats.pearsonr(valid[valid.side == "L"]["txt_mean"], valid[valid.side == "L"]["img_mean"])
    rho_l, _ = stats.spearmanr(valid[valid.side == "L"]["txt_mean"], valid[valid.side == "L"]["img_mean"])

    r_r, p_r = stats.pearsonr(valid[valid.side == "R"]["txt_mean"], valid[valid.side == "R"]["img_mean"])
    rho_r, _ = stats.spearmanr(valid[valid.side == "R"]["txt_mean"], valid[valid.side == "R"]["img_mean"])

    # ---- Token-Level Correlation (row-major) ----
    pooled_txt_all = []
    pooled_img_all = []
    per_sentence_rs = []
    for _, rec in df_records.iterrows():
        txt_scores = np.array(rec["txt_scores"])
        img_flat = np.array(rec["img_flat"])
        if len(txt_scores) < 2 or len(img_flat) < 2:
            continue
        L_target = max(2, min(len(txt_scores), len(img_flat)))
        txt_x = np.linspace(0, 1, len(txt_scores))
        img_x = np.linspace(0, 1, len(img_flat))
        new_x = np.linspace(0, 1, L_target)
        txt_interp = np.interp(new_x, txt_x, txt_scores)
        img_interp = np.interp(new_x, img_x, img_flat)
        pooled_txt_all.append(txt_interp)
        pooled_img_all.append(img_interp)
        r_sent, _ = stats.pearsonr(txt_interp, img_interp)
        per_sentence_rs.append({"pair_idx": rec["pair_idx"], "side": rec["side"], "r": r_sent})

    pooled_txt = np.concatenate(pooled_txt_all)
    pooled_img = np.concatenate(pooled_img_all)
    r_pooled, p_pooled = stats.pearsonr(pooled_txt, pooled_img)
    rho_pooled, p_rho_pooled = stats.spearmanr(pooled_txt, pooled_img)

    rs_df = pd.DataFrame(per_sentence_rs)
    per_sent_r_mean = rs_df["r"].mean()
    per_sent_r_median = rs_df["r"].median()

    # ---- Column-Level Correlation ----
    col_pooled_txt = []
    col_pooled_img = []
    col_per_sent_rs = []
    for _, rec in df_records.iterrows():
        txt_scores = np.array(rec["txt_scores"])
        if len(txt_scores) < 2 or rec["img_n_cells"] < 2:
            continue
        n_rows = rec["img_n_cells"] // gw
        if n_rows < 1:
            continue
        flat = np.array(rec["img_flat"])
        flat = flat[: n_rows * gw]
        if len(flat) < n_rows * gw:
            continue
        img_2d = flat.reshape(n_rows, gw)
        col_means = img_2d.mean(axis=0)
        L_target = max(2, min(len(txt_scores), len(col_means)))
        txt_x = np.linspace(0, 1, len(txt_scores))
        col_x = np.linspace(0, 1, len(col_means))
        new_x = np.linspace(0, 1, L_target)
        txt_interp = np.interp(new_x, txt_x, txt_scores)
        col_interp = np.interp(new_x, col_x, col_means)
        col_pooled_txt.append(txt_interp)
        col_pooled_img.append(col_interp)
        r_col, _ = stats.pearsonr(txt_interp, col_interp)
        col_per_sent_rs.append({"pair_idx": rec["pair_idx"], "side": rec["side"], "r": r_col})

    col_pooled_txt_arr = np.concatenate(col_pooled_txt) if col_pooled_txt else np.array([])
    col_pooled_img_arr = np.concatenate(col_pooled_img) if col_pooled_img else np.array([])
    col_rs_df = pd.DataFrame(col_per_sent_rs)

    if len(col_pooled_txt_arr) > 0:
        r_col_pooled, p_col_pooled = stats.pearsonr(col_pooled_txt_arr, col_pooled_img_arr)
        rho_col_pooled, _ = stats.spearmanr(col_pooled_txt_arr, col_pooled_img_arr)
        col_r_mean = col_rs_df["r"].mean()
    else:
        r_col_pooled = p_col_pooled = rho_col_pooled = col_r_mean = float("nan")

    # ---- Summary Table ----
    summary_rows = [
        {"Level": "Sentence (all)", "n": len(valid), "Pearson r": r_all, "p": p_all, "Spearman rho": rho_all},
        {
            "Level": "Sentence (L only)",
            "n": len(valid[valid.side == "L"]),
            "Pearson r": r_l,
            "p": p_l,
            "Spearman rho": rho_l,
        },
        {
            "Level": "Sentence (R only)",
            "n": len(valid[valid.side == "R"]),
            "Pearson r": r_r,
            "p": p_r,
            "Spearman rho": rho_r,
        },
        {"Level": "L-R diff", "n": PAIRS_PER_ITER, "Pearson r": np.nan, "p": np.nan, "Spearman rho": np.nan},
        {
            "Level": "Token (pooled, row-major)",
            "n": len(pooled_txt),
            "Pearson r": r_pooled,
            "p": p_pooled,
            "Spearman rho": rho_pooled,
        },
        {
            "Level": "Token (pooled, col-means)",
            "n": len(col_pooled_txt_arr),
            "Pearson r": r_col_pooled,
            "p": p_col_pooled,
            "Spearman rho": rho_col_pooled,
        },
        {
            "Level": "Per-sentence token r (mean)",
            "n": len(rs_df),
            "Pearson r": per_sent_r_mean,
            "p": np.nan,
            "Spearman rho": np.nan,
        },
        {
            "Level": "Per-sentence col r (mean)",
            "n": len(col_rs_df),
            "Pearson r": col_r_mean,
            "p": np.nan,
            "Spearman rho": np.nan,
        },
    ]
    summary = pd.DataFrame(summary_rows).round(4)

    # ---- Plot: Sentence-Level Correlation Scatter ----
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    palette = {"L": "#4E79A7", "R": "#E15759"}
    markers = {"L": "o", "R": "s"}

    for side in ["L", "R"]:
        sub = valid[valid["side"] == side]
        ax.scatter(
            sub["txt_mean"],
            sub["img_mean"],
            c=palette[side],
            marker=markers[side],
            s=22,
            edgecolors="white",
            linewidth=0.4,
            alpha=0.85,
            label=side,
            zorder=3,
        )

    ax.axhline(0, color="0.75", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(0, color="0.75", linewidth=0.6, linestyle="--", zorder=1)

    label_str = (
        f"Pearson r = {r_all:.2f}, p = {p_all:.2g}\n"
        f"Spearman \u03c1 = {rho_all:.2f}, p = {p_rho_all:.2g}"
    )
    ax.text(0.04, 0.96, label_str, transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_xlabel("Mean text token score")
    ax.set_ylabel("Mean image-region score")
    ax.set_title(f"{model_family}  top_k={top_k}  n={len(valid)}", pad=6)

    leg = ax.legend(frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.9", linewidth=0.4, zorder=0)
    plt.tight_layout()

    tag = f"{model_family}_k{top_k:02d}"
    plot_path = os.path.join(output_dir, f"scatter_{tag}.png")
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.savefig(plot_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")

    # ---- Save summary table ----
    csv_path = os.path.join(output_dir, f"summary_{tag}.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")

    # ---- Print summary ----
    print(summary.to_string(index=False))

    # Cleanup model to free GPU memory
    del model
    torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Image Canvas vs Direct Text Correlation")
    parser.add_argument("--model-families", nargs="+", default=["qwen3-vl", "gemma4"],
                        choices=["qwen3-vl", "gemma4"])
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Directory for saving plots and summary tables")
    parser.add_argument("--n-iters", type=int, default=50, help="Number of batch iterations")
    parser.add_argument("--pairs-per-iter", type=int, default=4, help="Pairs per iteration")
    args = parser.parse_args()

    global N_ITERS, PAIRS_PER_ITER
    N_ITERS = args.n_iters
    PAIRS_PER_ITER = args.pairs_per_iter

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    tv_df = pd.read_csv(TV_PATH)
    print(f"TwinViews: {len(tv_df)} pairs, {tv_df['topic'].nunique()} topics")

    # Load font
    font = load_font()
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + LINE_SPACING

    all_summaries = []
    for model_family in args.model_families:
        for top_k in args.top_k_values:
            summary = run_for_model_k(model_family, top_k, tv_df, font, line_h, args.output_dir)
            summary["model"] = model_family
            summary["top_k"] = top_k
            all_summaries.append(summary)

    # ---- Combined summary across all models/k ----
    combined_path = os.path.join(args.output_dir, "summary_all.csv")
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined summary saved to: {combined_path}")


if __name__ == "__main__":
    main()
