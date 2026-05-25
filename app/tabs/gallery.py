"""EasyPortrait Gallery tab — pre-computed heatmaps + generated news articles."""

import json
import os
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

from app import ROOT_DIR
from app.common import make_heatmap_overlay

EASYPORTRAIT_MODELS = ["gemma4", "qwen3_vl"]
EASYPORTRAIT_PROBES = [
    "combined_ideology_headwise_linear",
    "combined_ideology_layerwise_linear",
    "textual_ideology_headwise_linear",
    "textual_ideology_layerwise_linear",
]
EASYPORTRAIT_DEFAULT_PROBE = "combined_ideology_headwise_linear"
EASYPORTRAIT_LABELS = ["ALL", "DEM", "LEAN_DEM", "NEUTRAL", "LEAN_REP", "REP", "REFUSAL"]
_easyportrait_cache: Dict[str, Any] = {}
_gallery_masks_cache: Dict[str, Any] = {}

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


def _load_easyportrait_metadata(model: str, probe: str) -> pd.DataFrame:
    csv_path = os.path.join(
        ROOT_DIR, "results", "token_scoring", model, "easyportrait",
        f"prompt_token_stats_{probe}.csv",
    )
    jsonl_path = os.path.join(
        ROOT_DIR, "results", "prompt_generation", model,
        "easyportrait_1000samples_max512tok_features.jsonl",
    )

    tk_df = pd.read_csv(csv_path)
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    pg_df = pd.DataFrame(records)

    merge_cols = ["record_id", "political_label", "political_confidence", "political_score", "response", "extracted_label", "extracted_features"]
    return tk_df.merge(pg_df[merge_cols], on="record_id", how="inner")


def _ensure_gallery_masks_loaded(model: str) -> Dict[str, Any]:
    if model in _gallery_masks_cache:
        return _gallery_masks_cache[model]

    masks_npz_path = os.path.join(
        ROOT_DIR, "results", "prompt_generation", model,
        "easyportrait_1000samples_max512tok_features_sam3_masks.npz",
    )
    sam3_path = os.path.join(
        ROOT_DIR, "results", "prompt_generation", model,
        "easyportrait_1000samples_max512tok_features_sam3.jsonl",
    )

    masks = {}
    if os.path.exists(masks_npz_path):
        npz = np.load(masks_npz_path)
        masks = {k: npz[k] for k in npz.files}
        npz.close()

    mask_meta = {}
    if os.path.exists(sam3_path):
        with open(sam3_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec.get("record_id")
                fm = rec.get("feature_masks", {})
                if rid and fm:
                    mask_meta[rid] = list(fm.keys())

    data = {"masks": masks, "meta": mask_meta}
    _gallery_masks_cache[model] = data
    return data


def _boundary_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    dilated = binary_dilation(mask, iterations=thickness)
    return (dilated.astype(np.uint8) - mask.astype(np.uint8)).clip(0, 1)


def _overlay_boundaries(image: Image.Image, masks: list, features: list) -> Image.Image:
    """Draw colored boundary outlines for each mask onto the image."""
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

    # Legend
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
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


def _ensure_easyportrait_loaded(model: str, probe: str) -> Dict[str, Any]:
    cache_key = f"{model}:{probe}"
    if cache_key in _easyportrait_cache:
        return _easyportrait_cache[cache_key]

    df = _load_easyportrait_metadata(model, probe)
    npz_path = os.path.join(
        ROOT_DIR, "results", "token_scoring", model, "easyportrait",
        f"prompt_image_token_scores_{probe}.npz",
    )
    data = np.load(npz_path)
    record_scores = {rid: data[rid] for rid in data.files}
    data.close()

    global_vmin = -1.0
    global_vmax = 1.0

    _easyportrait_cache[cache_key] = {
        "df": df,
        "scores": record_scores,
        "vmin": float(global_vmin),
        "vmax": float(global_vmax),
    }
    return _easyportrait_cache[cache_key]


def _filter_records(model: str, probe: str, label: str) -> pd.DataFrame:
    data = _ensure_easyportrait_loaded(model, probe)
    df = data["df"].copy()
    if label != "ALL":
        df = df[df["extracted_label"] == label]
    return df


def _format_record_label(row: pd.Series) -> str:
    rid = str(row["record_id"])
    pol = row.get("extracted_label", "?")
    score = row.get("political_score", float("nan"))
    conf = int(row.get("political_confidence", 0))
    return f"{rid}  |  {pol}  |  score={score:+.1f}  |  conf={conf}/5"


def _easyportrait_text_info(row: pd.Series) -> str:
    label = row.get("extracted_label", "?")
    score = row.get("political_score", float("nan"))
    conf = int(row.get("political_confidence", 0))
    resp = str(row.get("response", "") or "")
    image_mean = float(row.get("image_mean", float("nan")))
    fg_mean = float(row.get("image_fg_mean", float("nan")))
    bg_mean = float(row.get("image_bg_mean", float("nan")))
    last_score = float(row.get("all_last_token_score", float("nan")))
    features = row.get("extracted_features", None)
    if isinstance(features, str):
        import json as _json
        try:
            features = _json.loads(features)
        except Exception:
            features = [features]

    display_text = resp[:1200] + ("..." if len(resp) > 1200 else "")
    features_str = ", ".join(features) if features else "none extracted"
    return (
        f"### {row.get('record_id', '?')}\n\n"
        f"**Political:** {label}  |  score = {score:+.2f}  |  conf = {conf}/5\n\n"
        f"**Extracted features:** {features_str}\n\n"
        f"**Image token score mean:** {image_mean:.4f}\n"
        f"**Foreground mean / Background mean:** {fg_mean:.4f} / {bg_mean:.4f}\n"
        f"**Last prompt token score:** {last_score:.4f}\n\n"
        f"---\n\n{display_text}"
    )


def easyportrait_show_record(
    model: str, probe: str, label: str, record_choice: str,
) -> Tuple[Optional[Image.Image], str]:
    if not model or not record_choice:
        return None, ""
    if record_choice.startswith("No records"):
        return None, ""

    record_id = record_choice.split("  |")[0].strip()
    data = _ensure_easyportrait_loaded(model, probe)
    df = data["df"]

    mask = df["record_id"].astype(str) == record_id
    if not mask.any():
        return None, f"Record {record_id} not found."
    row = df[mask].iloc[0]

    if record_id not in data["scores"]:
        return None, f"No token scores for {record_id}."

    scores = data["scores"][record_id]
    grid_h = int(row.get("grid_h", -1))
    grid_w = int(row.get("grid_w", -1))

    img_path = os.path.join(ROOT_DIR, str(row.get("image_path", "")))
    if not img_path or not os.path.exists(img_path):
        return None, "Image not found"

    pil_image = Image.open(img_path).convert("L").convert("RGB")
    orig_h, orig_w = pil_image.height, pil_image.width

    if grid_h > 0 and grid_w > 0:
        expected = grid_h * grid_w
        if len(scores) > expected:
            scores = scores[:expected]
        elif len(scores) < expected:
            scores = np.concatenate([scores, np.zeros(expected - len(scores), dtype=scores.dtype)])
    else:
        return pil_image, _easyportrait_text_info(row)

    overlay = make_heatmap_overlay(
        pil_image, scores, (grid_h, grid_w), (orig_h, orig_w),
        vmin=data["vmin"], vmax=data["vmax"],
    )

    # Overlay SAM3 feature mask boundaries
    mask_data = _ensure_gallery_masks_loaded(model)
    features = mask_data["meta"].get(record_id, [])
    masks_list = []
    for fname in features:
        mk = f"{record_id}/{fname}"
        m = mask_data["masks"].get(mk)
        if m is not None:
            masks_list.append(m)
    if masks_list:
        overlay = _overlay_boundaries(overlay, masks_list, features)

    return overlay, _easyportrait_text_info(row)


def easyportrait_update_model_or_label(
    model: str, probe: str, label: str,
) -> Tuple[gr.Dropdown, Optional[Image.Image], str]:
    """Refresh record list for model/probe/label and show first record."""
    if not model:
        return gr.Dropdown(choices=[]), None, ""

    df = _filter_records(model, probe, label)
    choices = [_format_record_label(row) for _, row in df.iterrows()]

    if not choices:
        return gr.Dropdown(choices=[]), None, "No records match the filter."

    first_choice = choices[0]
    heatmap, text = easyportrait_show_record(model, probe, label, first_choice)
    return gr.Dropdown(choices=choices, value=first_choice), heatmap, text


def _build_gallery_tab():
    """Build the EasyPortrait gallery tab UI."""
    gr.Markdown(
        "# EasyPortrait: Image Token Scores + Political Leaning\n"
        "Browse pre-computed token-score heatmaps and LLM-generated political "
        "leaning guesses for EasyPortrait images. "
        "Supporting visual features extracted from each response are shown. "
        "Select a model and filter by political label."
    )

    model_dropdown = gr.Dropdown(
        EASYPORTRAIT_MODELS, value=EASYPORTRAIT_MODELS[0], label="Model",
    )
    probe_dropdown = gr.Dropdown(
        EASYPORTRAIT_PROBES, value=EASYPORTRAIT_DEFAULT_PROBE, label="Probe",
    )
    label_dropdown = gr.Dropdown(
        EASYPORTRAIT_LABELS, value="ALL", label="Political Label Filter",
    )
    record_dropdown = gr.Dropdown(
        choices=[], label="Select Record", interactive=True,
    )

    with gr.Row():
        with gr.Column(scale=1):
            gallery_heatmap = gr.Image(type="pil", label="Token Score Heatmap")
        with gr.Column(scale=1):
            gallery_text = gr.Markdown("Select a record to view details.")

    gallery_inputs = [model_dropdown, probe_dropdown, label_dropdown, record_dropdown]
    updater_inputs = [model_dropdown, probe_dropdown, label_dropdown]

    model_dropdown.change(
        fn=easyportrait_update_model_or_label,
        inputs=updater_inputs,
        outputs=[record_dropdown, gallery_heatmap, gallery_text],
    )
    probe_dropdown.change(
        fn=easyportrait_update_model_or_label,
        inputs=updater_inputs,
        outputs=[record_dropdown, gallery_heatmap, gallery_text],
    )
    label_dropdown.change(
        fn=easyportrait_update_model_or_label,
        inputs=updater_inputs,
        outputs=[record_dropdown, gallery_heatmap, gallery_text],
    )
    record_dropdown.change(
        fn=easyportrait_show_record,
        inputs=gallery_inputs,
        outputs=[gallery_heatmap, gallery_text],
    )
