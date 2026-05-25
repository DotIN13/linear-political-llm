"""Feature masks tab — visualize combined SAM3 segmentation masks per record."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

from app import ROOT_DIR

FEATURE_MASKS_MODELS = ["gemma4", "qwen3_vl"]
DEFAULT_PROBE = "combined_ideology_headwise_linear"
_feature_masks_cache: Dict[str, Any] = {}

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


def _vis_dir(model: str) -> str:
    return os.path.join(
        ROOT_DIR, "results", "prompt_generation", model,
        "easyportrait_1000samples_max512tok_features_sam3_vis",
    )


def _vis_path(model: str, record_id: str) -> str:
    return os.path.join(_vis_dir(model), f"{record_id}.png")


def _boundary_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    dilated = binary_dilation(mask, iterations=thickness)
    return (dilated.astype(np.uint8) - mask.astype(np.uint8)).clip(0, 1)


def _render_combined_masks(image: Image.Image, masks: list, features: list) -> Image.Image:
    """Overlay mask boundaries with distinct colors and a legend."""
    image = image.convert("RGBA")
    for i, mask in enumerate(masks):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        boundary = _boundary_mask(mask)
        boundary_img = Image.fromarray((boundary * 255).astype(np.uint8))
        boundary_img = boundary_img.resize(image.size, Image.NEAREST)
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(boundary_img)
        image = Image.alpha_composite(image, overlay)
    result = image.convert("RGB")

    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    y = 6
    for i, fname in enumerate(features):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        bbox = draw.textbbox((0, 0), fname, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 5
        draw.rectangle([4, y, 4 + tw + pad * 2 + 20, y + th + pad], fill=(0, 0, 0, 180))
        draw.rectangle([8, y + 3, 20, y + th + pad - 3], fill=color)
        draw.text((26, y + pad // 2), fname, fill=(255, 255, 255), font=font)
        y += th + pad + 4
    return result


def _load_sam3_image_lookup(model: str) -> Dict[str, str]:
    csv_path = os.path.join(
        ROOT_DIR, "results", "token_scoring", model, "easyportrait",
        f"prompt_token_fg_bg_stats_{DEFAULT_PROBE}.csv",
    )
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path, usecols=["record_id", "image_path"])
    return {str(row["record_id"]): os.path.join(ROOT_DIR, str(row["image_path"])) for _, row in df.iterrows()}


def _ensure_feature_masks_loaded(model: str) -> Dict[str, Any]:
    if model in _feature_masks_cache:
        return _feature_masks_cache[model]

    sam3_path = os.path.join(
        ROOT_DIR, "results", "prompt_generation", model,
        "easyportrait_1000samples_max512tok_features_sam3.jsonl",
    )
    masks_npz_path = os.path.join(
        ROOT_DIR, "results", "prompt_generation", model,
        "easyportrait_1000samples_max512tok_features_sam3_masks.npz",
    )

    records = []
    if os.path.exists(sam3_path):
        with open(sam3_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    masks = {}
    if os.path.exists(masks_npz_path):
        npz = np.load(masks_npz_path)
        masks = {k: npz[k] for k in npz.files}
        npz.close()

    image_lookup = _load_sam3_image_lookup(model)

    data = {
        "records": records,
        "masks": masks,
        "image_lookup": image_lookup,
    }
    _feature_masks_cache[model] = data
    return data


def _records_with_masks(model: str) -> List[Dict[str, Any]]:
    data = _ensure_feature_masks_loaded(model)
    return [r for r in data["records"] if r.get("feature_masks")]


def _format_record_choice(rec: Dict[str, Any]) -> str:
    rid = rec.get("record_id", "?")
    n_masks = len(rec.get("feature_masks", {}))
    features = list(rec.get("feature_masks", {}).keys())
    feat_preview = ", ".join(features[:3])
    if len(features) > 3:
        feat_preview += f", +{len(features) - 3} more"
    return f"{rid}  |  {n_masks} masks: {feat_preview}"


def feature_masks_update_records(model: str) -> Tuple[gr.Dropdown, Optional[Image.Image], str]:
    if not model:
        return gr.Dropdown(choices=[]), None, ""

    recs = _records_with_masks(model)
    choices = [_format_record_choice(r) for r in recs]

    if not choices:
        return gr.Dropdown(choices=[]), None, "No records with feature masks found."

    first_choice = choices[0]
    img, text = feature_masks_show_record(model, first_choice)
    return gr.Dropdown(choices=choices, value=first_choice), img, text


def feature_masks_show_record(
    model: str, record_choice: str,
) -> Tuple[Optional[Image.Image], str]:
    if not model or not record_choice:
        return None, ""

    rid = record_choice.split("  |")[0].strip()
    data = _ensure_feature_masks_loaded(model)

    image_path = data["image_lookup"].get(rid)
    if not image_path or not os.path.exists(image_path):
        return None, f"Image not found for {rid}"

    rec = next((r for r in data["records"] if r.get("record_id") == rid), None)
    if rec is None:
        return None, f"Record {rid} not found."

    feature_masks = rec.get("feature_masks", {})
    if not feature_masks:
        return None, f"No feature masks for {rid}"

    features = list(feature_masks.keys())
    label = rec.get("extracted_label", "?")
    score = rec.get("political_score", float("nan"))
    conf = rec.get("political_confidence", 0)
    text = (
        f"### {rid}\n\n"
        f"**Political:** {label}  |  score = {score:+.2f}  |  conf = {conf}/5\n\n"
        f"**Features with masks:** {', '.join(features)}\n"
    )

    # Try pre-generated combined vis image first
    vis_png = _vis_path(model, rid)
    if os.path.exists(vis_png):
        result = Image.open(vis_png).convert("RGB")
        for fname, meta in feature_masks.items():
            shape = meta.get("shape", [0, 0])
            mask_key = f"{rid}/{fname}"
            mask = data["masks"].get(mask_key)
            cov = mask.sum() / mask.size * 100 if mask is not None else 0
            text += f"\n**{fname}:** {shape[0]}x{shape[1]} ({cov:.1f}%)"
        return result, text

    # Generate on-the-fly and save for future use
    masks_list = []
    for fname in features:
        mask_key = f"{rid}/{fname}"
        mask = data["masks"].get(mask_key)
        if mask is None:
            continue
        masks_list.append(mask)
        shape = feature_masks[fname].get("shape", [0, 0])
        cov = mask.sum() / mask.size * 100 if mask is not None else 0
        text += f"\n**{fname}:** {shape[0]}x{shape[1]} ({cov:.1f}%)"

    if not masks_list:
        pil_image = Image.open(image_path).convert("RGB")
        return pil_image, text

    pil_image = Image.open(image_path).convert("RGB")
    first_mask = masks_list[0]
    pil_image = pil_image.resize((first_mask.shape[1], first_mask.shape[0]), Image.LANCZOS)
    result = _render_combined_masks(pil_image, masks_list, features)
    os.makedirs(os.path.dirname(vis_png), exist_ok=True)
    result.save(vis_png)
    return result, text


def _build_feature_masks_tab():
    """Build the Feature Masks visualization tab."""
    gr.Markdown(
        "# Feature Masks: SAM3 Segmentation\n"
        "Visualize SAM3 Promptable Concept Segmentation masks for visual features "
        "extracted from LLM responses. All features are overlaid on a single image "
        "with distinct colors and a legend."
    )

    model_dropdown = gr.Dropdown(
        FEATURE_MASKS_MODELS, value=FEATURE_MASKS_MODELS[0], label="Model",
    )
    record_dropdown = gr.Dropdown(
        choices=[], label="Select Record", interactive=True,
    )

    with gr.Row():
        with gr.Column(scale=1):
            mask_image = gr.Image(type="pil", label="Feature Mask Overlay")
        with gr.Column(scale=1):
            mask_text = gr.Markdown("Select a record to view all feature masks overlaid.")

    model_dropdown.change(
        fn=feature_masks_update_records,
        inputs=[model_dropdown],
        outputs=[record_dropdown, mask_image, mask_text],
    )
    record_dropdown.change(
        fn=feature_masks_show_record,
        inputs=[model_dropdown, record_dropdown],
        outputs=[mask_image, mask_text],
    )
