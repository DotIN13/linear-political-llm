"""
Score Congress portrait image tokens with Qwen3-VL for one or more ridge directions.

Example:
python scripts/run_qwen3_vl_congress_ideology_tokens.py --ridge-prefixes combined_ideology politician --top-k 16
"""

import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageOps
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lpl.utils import generate_and_score_tokens, model_base_name

DEFAULT_MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
DEFAULT_IMAGE_DIR = "data/congress_images"
DEFAULT_HS_PATH = "data/HS116_members.csv"
DEFAULT_CUR_PATH = "data/legislators-current.json"
DEFAULT_HIST_PATH = "data/legislators-historical.json"
DEFAULT_PROMPT = "What's his/her position on US politics?"
DEFAULT_RIDGE_PREFIXES = ["combined_ideology", "textual_ideology"]
DEFAULT_TOP_K = 16
DEFAULT_MAX_NEW_TOKENS = 1
DEFAULT_RESIZED_IMAGE_WIDTH = 250
PARTY_MAP = {100: "Democrat", 200: "Republican"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Congress portraits with Qwen3-VL token-level ideology directions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--hs-path", default=DEFAULT_HS_PATH)
    parser.add_argument("--current-legislators-path", default=DEFAULT_CUR_PATH)
    parser.add_argument("--historical-legislators-path", default=DEFAULT_HIST_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ridge-prefixes", nargs="+", default=DEFAULT_RIDGE_PREFIXES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--resized-image-width", type=int, default=DEFAULT_RESIZED_IMAGE_WIDTH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_dir(model_path: str, output_dir: str = None) -> str:
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    resolved = os.path.join("results", model_base_name(model_path))
    os.makedirs(resolved, exist_ok=True)
    return resolved


def resolve_output_paths(output_dir: str, ridge_prefix: str) -> Tuple[str, str, str, str]:
    activations_name = f"congress_images_patch_scores_{ridge_prefix}.pt"
    stats_name = f"congress_images_patch_stats_{ridge_prefix}.csv"
    summary_name = f"congress_images_patch_summary_{ridge_prefix}.png"
    correlation_name = f"congress_images_correlation_{ridge_prefix}.png"
    return (
        os.path.join(output_dir, activations_name),
        os.path.join(output_dir, stats_name),
        os.path.join(output_dir, summary_name),
        os.path.join(output_dir, correlation_name),
    )


def ensure_writable(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def load_legislator_name_map(paths: Sequence[str]) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)

        for record in records:
            bioguide = record.get("id", {}).get("bioguide")
            if not bioguide:
                continue

            official_full = record.get("name", {}).get("official_full")
            if official_full:
                names[bioguide] = official_full
                continue

            name_obj = record.get("name", {})
            first = name_obj.get("first", "")
            last = name_obj.get("last", "")
            fallback = " ".join(part for part in [first, last] if part).strip()
            if fallback:
                names[bioguide] = fallback

    return names


def build_probe_dataframe(
    image_dir: str,
    hs_path: str,
    current_legislators_path: str,
    historical_legislators_path: str,
) -> pd.DataFrame:
    name_map = load_legislator_name_map([current_legislators_path, historical_legislators_path])

    df_hs = pd.read_csv(hs_path)
    df_hs = df_hs[pd.notnull(df_hs["nominate_dim1"])].copy()
    df_hs["bioguide"] = df_hs["bioguide_id"].astype(str).str.strip().str.upper()
    df_hs["image_path"] = df_hs["bioguide"].apply(lambda bg: str(Path(image_dir) / f"{bg}.jpg"))
    df_hs["has_image"] = df_hs["image_path"].apply(os.path.exists)
    df_hs["name"] = df_hs["bioguide"].map(name_map).fillna(df_hs["bioname"])

    df_probe = df_hs[df_hs["has_image"]].copy().reset_index(drop=True)
    return df_probe


def vision_grid_hw(orig_h: int, orig_w: int, processor) -> Tuple[int, int]:
    image_processor = processor.image_processor
    min_pixels = image_processor.size["shortest_edge"]
    max_pixels = image_processor.size["longest_edge"]
    patch = image_processor.patch_size
    merge = image_processor.merge_size

    factor = patch * merge
    resized_h, resized_w = smart_resize(orig_h, orig_w, factor, min_pixels=min_pixels, max_pixels=max_pixels)
    grid_h = (resized_h // patch) // merge
    grid_w = (resized_w // patch) // merge
    return int(grid_h), int(grid_w)


def try_get_image_pad_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise RuntimeError("Could not find image pad token id in tokenizer.")
    return int(token_id)


def extract_image_token_scores(token_ids, scores, tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    token_ids_np = np.asarray(token_ids).squeeze()
    scores_np = np.asarray(scores).squeeze()

    if len(token_ids_np) == len(scores_np) + 1:
        token_ids_np = token_ids_np[:-1]

    image_pad_id = try_get_image_pad_id(tokenizer)
    image_idx = np.where(token_ids_np == image_pad_id)[0]
    return scores_np[image_idx], image_idx


def load_resized_portrait(image_path: str, target_width: int) -> Image.Image:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width > target_width:
            target_height = max(1, round(image.height * target_width / image.width))
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return image


def build_prompt(processor, image: Image.Image, prompt_text: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def build_portrait_inputs(
    df_probe: pd.DataFrame,
    processor,
    prompt_text: str,
    resized_image_width: int,
    limit: int = None,
) -> List[Dict]:
    rows = list(df_probe.itertuples(index=False))
    if limit is not None:
        rows = rows[:limit]

    records: List[Dict] = []
    for row in rows:
        image = load_resized_portrait(row.image_path, resized_image_width)
        grid_h, grid_w = vision_grid_hw(image.height, image.width, processor)
        records.append(
            {
                "bioguide": row.bioguide,
                "name": row.name,
                "party_code": int(row.party_code),
                "nominate_dim1": float(row.nominate_dim1),
                "image_path": row.image_path,
                "resized_width": int(image.width),
                "resized_height": int(image.height),
                "grid_shape": (grid_h, grid_w),
                "prompt": build_prompt(processor, image, prompt_text),
            }
        )

    return records


def score_direction(
    model_path: str,
    model,
    processor,
    portrait_inputs: List[Dict],
    ridge_prefix: str,
    top_k: int,
    max_new_tokens: int,
    device: torch.device,
):
    results = generate_and_score_tokens(
        model_path=model_path,
        prompts=[record["prompt"] for record in portrait_inputs],
        model=model,
        tokenizer=processor.tokenizer,
        max_new_tokens=max_new_tokens,
        k=top_k,
        ridge_prefix=ridge_prefix,
        output_only=False,
        visualize=False,
        mode="vision",
        device=device,
    )

    portrait_token_maps: List[Dict] = []
    summary_rows: List[Dict] = []
    skipped = 0

    for record, result in zip(portrait_inputs, results):
        grid_h, grid_w = record["grid_shape"]
        expected_tokens = grid_h * grid_w

        image_scores, image_token_idx = extract_image_token_scores(
            result["token_ids"],
            result["scores"],
            processor.tokenizer,
        )

        if len(image_scores) != expected_tokens:
            skipped += 1
            continue

        party = PARTY_MAP.get(record["party_code"], f"Other ({record['party_code']})")
        heatmap = image_scores.reshape(grid_h, grid_w)

        portrait_token_maps.append(
            {
                "bioguide": record["bioguide"],
                "name": record["name"],
                "party": party,
                "party_code": record["party_code"],
                "nominate_dim1": record["nominate_dim1"],
                "image_path": record["image_path"],
                "resized_width": record["resized_width"],
                "resized_height": record["resized_height"],
                "grid_shape": (grid_h, grid_w),
                "num_image_tokens": int(expected_tokens),
                "image_token_indices": image_token_idx,
                "image_token_scores": image_scores,
                "heatmap": heatmap,
            }
        )

        summary_rows.append(
            {
                "bioguide": record["bioguide"],
                "name": record["name"],
                "party": party,
                "party_code": record["party_code"],
                "nominate_dim1": record["nominate_dim1"],
                "resized_width": record["resized_width"],
                "resized_height": record["resized_height"],
                "grid_h": int(grid_h),
                "grid_w": int(grid_w),
                "num_image_tokens": int(expected_tokens),
                "mean_image_token_score": float(np.mean(image_scores)),
                "median_image_token_score": float(np.median(image_scores)),
                "mean_abs_image_token_score": float(np.mean(np.abs(image_scores))),
                "min_image_token_score": float(np.min(image_scores)),
                "max_image_token_score": float(np.max(image_scores)),
            }
        )

    return pd.DataFrame(summary_rows), portrait_token_maps, skipped


def save_pt_output(path: str, portrait_token_maps: List[Dict], metadata: Dict[str, object]) -> None:
    image_paths: List[str] = []
    image_names: List[str] = []
    bioguides: List[str] = []
    names: List[str] = []
    parties: List[str] = []
    grid_hw: List[Tuple[int, int]] = []
    offsets: List[int] = [0]
    flat_scores: List[torch.Tensor] = []

    for record in portrait_token_maps:
        scores = np.asarray(record["image_token_scores"], dtype=np.float32)
        image_paths.append(record["image_path"])
        image_names.append(os.path.basename(record["image_path"]))
        bioguides.append(record["bioguide"])
        names.append(record["name"])
        parties.append(record["party"])
        grid_h, grid_w = record["grid_shape"]
        grid_hw.append((int(grid_h), int(grid_w)))
        flat_scores.append(torch.from_numpy(scores.copy()))
        offsets.append(offsets[-1] + int(scores.size))

    score_tensor = torch.cat(flat_scores, dim=0) if flat_scores else torch.empty(0, dtype=torch.float32)
    payload = {
        "metadata": metadata,
        "image_paths": image_paths,
        "image_names": image_names,
        "bioguides": bioguides,
        "names": names,
        "parties": parties,
        "grid_hw": torch.tensor(grid_hw, dtype=torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "scores": score_tensor,
    }
    torch.save(payload, path)


def save_correlation_image(summary_df: pd.DataFrame, output_path: str, ridge_prefix: str) -> None:
    corr_input = summary_df.dropna(subset=["nominate_dim1", "mean_image_token_score"]).copy()
    if len(corr_input) < 3:
        return

    overall_pearson = float(pearsonr(corr_input["nominate_dim1"], corr_input["mean_image_token_score"]).statistic)
    overall_spearman = float(spearmanr(corr_input["nominate_dim1"], corr_input["mean_image_token_score"]).statistic)

    party_palette = {"Democrat": "#1f77b4", "Republican": "#d62728", "Other (328)": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, constrained_layout=True)

    # Panel 1: pooled scatter + pooled regression line
    sns.scatterplot(
        data=corr_input,
        x="nominate_dim1",
        y="mean_image_token_score",
        hue="party",
        palette=party_palette,
        alpha=0.75,
        s=45,
        ax=axes[0],
    )
    sns.regplot(
        data=corr_input,
        x="nominate_dim1",
        y="mean_image_token_score",
        scatter=False,
        color="black",
        line_kws={"linewidth": 2.0},
        ax=axes[0],
    )
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_title(
        f"Pooled correlation\nPearson r={overall_pearson:.3f}, Spearman rho={overall_spearman:.3f}"
    )
    axes[0].set_xlabel("DW-NOMINATE dim1")
    axes[0].set_ylabel("Mean image-token score")

    # Panel 2: party-specific regressions
    for party, grp in corr_input.groupby("party"):
        color = party_palette.get(party, None)
        if len(grp) >= 2:
            sns.regplot(
                data=grp,
                x="nominate_dim1",
                y="mean_image_token_score",
                scatter=True,
                ci=None,
                scatter_kws={"alpha": 0.55, "s": 35},
                line_kws={"linewidth": 2.2},
                color=color,
                ax=axes[1],
                label=party,
            )
        else:
            axes[1].scatter(grp["nominate_dim1"], grp["mean_image_token_score"], color=color, alpha=0.7, s=35, label=party)

    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_title("Party-specific regression lines")
    axes[1].set_xlabel("DW-NOMINATE dim1")
    axes[1].set_ylabel("Mean image-token score")
    axes[1].legend(loc="best", title="party")

    fig.suptitle(f"DW-NOMINATE correlation ({ridge_prefix})")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_summary_image(summary_df: pd.DataFrame, portrait_token_maps: List[Dict], output_path: str, ridge_prefix: str) -> None:
    party_order = [party for party in ["Democrat", "Republican"] if party in summary_df["party"].unique()]
    plot_df = summary_df[summary_df["party"].isin(party_order)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150, constrained_layout=True)
    axes = axes.flatten()

    if plot_df.empty:
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.5, 0.5, "No Democrat/Republican portraits were scored.", ha="center", va="center")
        fig.suptitle(f"Congress token score summary ({ridge_prefix})")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    common_grid_counts = Counter(
        record["grid_shape"]
        for record in portrait_token_maps
        if record["party"] in party_order
    )

    if len(common_grid_counts) > 0:
        common_grid, common_grid_count = common_grid_counts.most_common(1)[0]
        heatmap_records = [
            record
            for record in portrait_token_maps
            if record["party"] in party_order and record["grid_shape"] == common_grid
        ]
    else:
        common_grid, common_grid_count = (0, 0), 0
        heatmap_records = []

    party_heatmaps = {
        party: np.stack([record["heatmap"] for record in heatmap_records if record["party"] == party])
        for party in party_order
    }
    party_heatmaps = {
        party: values.mean(axis=0)
        for party, values in party_heatmaps.items()
        if len(values) > 0
    }

    heatmap_values = list(party_heatmaps.values())
    party_palette = {"Democrat": "#1f77b4", "Republican": "#d62728"}

    if len(heatmap_values) == 0:
        for ax in axes[:3]:
            ax.axis("off")
        axes[0].text(0.5, 0.5, "No common-grid heatmap could be built.", ha="center", va="center")
    else:
        heatmap_limit = max(float(np.abs(heatmap).max()) for heatmap in heatmap_values)
        for ax_idx, party in enumerate(party_order[:2]):
            if party not in party_heatmaps:
                axes[ax_idx].axis("off")
                continue
            ax = axes[ax_idx]
            sns.heatmap(
                party_heatmaps[party],
                ax=ax,
                cmap="RdBu_r",
                center=0,
                vmin=-heatmap_limit,
                vmax=heatmap_limit,
                cbar_kws={"label": "Mean image-token score"},
            )
            ax.set_title(f"{party} mean image-token score\n{common_grid_count} portraits share grid {common_grid}")
            ax.set_xlabel("Image token column")
            ax.set_ylabel("Image token row")

        if len(party_heatmaps) == 2 and "Democrat" in party_heatmaps and "Republican" in party_heatmaps:
            diff_heatmap = party_heatmaps["Republican"] - party_heatmaps["Democrat"]
            diff_limit = float(np.abs(diff_heatmap).max())
            sns.heatmap(
                diff_heatmap,
                ax=axes[2],
                cmap="RdBu_r",
                center=0,
                vmin=-diff_limit,
                vmax=diff_limit,
                cbar_kws={"label": "Republican - Democrat"},
            )
            axes[2].set_title("Party difference in mean image-token score")
            axes[2].set_xlabel("Image token column")
            axes[2].set_ylabel("Image token row")
        else:
            axes[2].axis("off")

    sns.boxplot(
        data=plot_df,
        x="party",
        y="mean_image_token_score",
        order=party_order,
        palette=party_palette,
        ax=axes[3],
    )
    sns.stripplot(
        data=plot_df,
        x="party",
        y="mean_image_token_score",
        order=party_order,
        color="gray",
        alpha=0.35,
        size=3,
        ax=axes[3],
    )
    axes[3].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[3].set_title("Per-portrait mean image-token score by party")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("Mean image-token score")

    fig.suptitle(f"Congress token score summary ({ridge_prefix})")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.model_path, args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and processor...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    print("Building Congress probe dataframe...")
    df_probe = build_probe_dataframe(
        image_dir=args.image_dir,
        hs_path=args.hs_path,
        current_legislators_path=args.current_legislators_path,
        historical_legislators_path=args.historical_legislators_path,
    )
    print(f"Rows with portraits: {len(df_probe)}")

    portrait_inputs = build_portrait_inputs(
        df_probe=df_probe,
        processor=processor,
        prompt_text=args.prompt,
        resized_image_width=args.resized_image_width,
        limit=args.limit,
    )
    if not portrait_inputs:
        raise RuntimeError("No portrait inputs were built. Check input data paths.")

    print(f"Prepared portrait prompts: {len(portrait_inputs)}")

    for ridge_prefix in args.ridge_prefixes:
        activations_path, stats_path, summary_path, correlation_path = resolve_output_paths(output_dir, ridge_prefix)
        ensure_writable(activations_path, args.overwrite)
        ensure_writable(stats_path, args.overwrite)
        ensure_writable(summary_path, args.overwrite)
        ensure_writable(correlation_path, args.overwrite)

        print(f"Scoring ridge direction: {ridge_prefix}")
        summary_df, portrait_token_maps, skipped = score_direction(
            model_path=args.model_path,
            model=model,
            processor=processor,
            portrait_inputs=portrait_inputs,
            ridge_prefix=ridge_prefix,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        metadata = {
            "model_path": args.model_path,
            "image_dir": args.image_dir,
            "ridge_prefix": ridge_prefix,
            "top_k": int(args.top_k),
            "prompt": args.prompt,
            "resized_image_width": int(args.resized_image_width),
            "max_new_tokens": int(args.max_new_tokens),
            "num_portraits_input": int(len(portrait_inputs)),
            "num_portraits_scored": int(len(summary_df)),
            "num_skipped_mismatch": int(skipped),
        }

        save_pt_output(activations_path, portrait_token_maps, metadata)
        summary_df.to_csv(stats_path, index=False)
        save_summary_image(summary_df, portrait_token_maps, summary_path, ridge_prefix)
        save_correlation_image(summary_df, correlation_path, ridge_prefix)

        print(f"Saved activations PT: {activations_path}")
        print(f"Saved stats CSV: {stats_path}")
        print(f"Saved summary image: {summary_path}")
        print(f"Saved correlation image: {correlation_path}")
        print(f"Scored portraits: {len(summary_df)} | skipped mismatch: {skipped}")
        if not summary_df.empty:
            print("Party counts:")
            print(summary_df["party"].value_counts().to_string())


if __name__ == "__main__":
    main()
