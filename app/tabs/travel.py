"""Travel Recommendations tab — image + mean token score + recommended cities/states."""

import json
import os
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import pandas as pd
from PIL import Image

from app import ROOT_DIR

TRAVEL_MODELS = ["qwen3_vl", "gemma4"]
TRAVEL_LABELS = ["ALL", "LEAN_DEM", "LEAN_REP"]
_travel_cache: Dict[str, Any] = {}


def _load_travel_data(model: str) -> pd.DataFrame:
    cache_key = f"travel:{model}"
    if cache_key in _travel_cache:
        return _travel_cache[cache_key]

    jsonl_path = os.path.join(
        ROOT_DIR, "results", "prompt_generation", model,
        "easyportrait_1000samples_max512tok_features_travel_responses.jsonl",
    )
    records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    df = pd.DataFrame(records)
    _travel_cache[cache_key] = df
    return df


def _filter_travel_records(model: str, label: str) -> pd.DataFrame:
    df = _load_travel_data(model)
    if df.empty:
        return df
    successful = df[df["_travel_response"].notna() & (df["_travel_response"] != "")]
    if label != "ALL":
        successful = successful[successful["extracted_label"] == label]
    return successful


def _format_travel_label(row: pd.Series) -> str:
    rid = str(row.get("record_id", "?"))
    pol = row.get("extracted_label", "?")
    score = row.get("all_mean", None)
    score_str = f"score={score:.4f}" if score is not None else "score=N/A"
    locations = row.get("_locations", [])
    if locations:
        states = sorted(set(loc.get("state", "") for loc in locations if loc.get("state")))
        cities = [f"{loc['city']}, {loc['state']}" for loc in locations if loc.get("city")]
        cities_unique = sorted(set(cities))
        loc_str = " | ".join(cities_unique[:3])
        if len(cities_unique) > 3:
            loc_str += f" (+{len(cities_unique) - 3} more)"
    else:
        loc_str = "no locations found"
    return f"{rid}  |  {pol}  |  {score_str}  |  {loc_str}"


def _travel_text_info(row: pd.Series) -> str:
    rid = str(row.get("record_id", "?"))
    label = row.get("extracted_label", "?")
    score = row.get("all_mean", None)
    response = str(row.get("_travel_response", "") or "")
    features = row.get("extracted_features", [])
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except Exception:
            features = [features]
    features_str = ", ".join(features) if features else "none extracted"

    locations = row.get("_locations", [])
    if locations:
        cities_lines = []
        for loc in locations:
            city = loc.get("city", "")
            state = loc.get("state", "")
            if city:
                cities_lines.append(f"- **{city}**, {state}" if state else f"- **{city}**")
        loc_str = "\n".join(sorted(set(cities_lines))) if cities_lines else "_no cities detected_"
    else:
        loc_str = "_no locations extracted_"

    display_response = response[:2000] + ("..." if len(response) > 2000 else "")

    score_line = f"**Mean token score (all_mean):** {score:.4f}\n\n" if score is not None else "**Mean token score:** N/A\n\n"

    return (
        f"### {rid}\n\n"
        f"**Political label:** {label}\n\n"
        f"{score_line}"
        f"**Extracted features:** {features_str}\n\n"
        f"**Recommended cities/states:**\n{loc_str}\n\n"
        f"---\n\n{display_response}"
    )


def travel_show_record(
    model: str, label: str, record_choice: str,
) -> Tuple[Optional[Image.Image], str]:
    if not model or not record_choice:
        return None, ""
    if record_choice.startswith("No records"):
        return None, ""

    record_id = record_choice.split("  |")[0].strip()
    df = _filter_travel_records(model, label)
    if df.empty:
        return None, "No matching records."

    mask = df["record_id"].astype(str) == record_id
    if not mask.any():
        return None, f"Record {record_id} not found."
    row = df[mask].iloc[0]

    img_path = str(row.get("image_path", ""))
    if not img_path or not os.path.exists(img_path):
        return None, f"Image not found: {img_path}"

    image = Image.open(img_path).convert("RGB")
    return image, _travel_text_info(row)


def travel_update_model_or_label(
    model: str, label: str,
) -> Tuple[gr.Dropdown, Optional[Image.Image], str]:
    if not model:
        return gr.Dropdown(choices=[]), None, ""

    df = _filter_travel_records(model, label)
    if df.empty:
        return gr.Dropdown(choices=["No records for this model/label"]), None, ""

    choices = [_format_travel_label(row) for _, row in df.iterrows()]
    first_choice = choices[0]
    image, text = travel_show_record(model, label, first_choice)
    return gr.Dropdown(choices=choices, value=first_choice), image, text


def _build_travel_tab():
    """Build the Travel Recommendations tab UI."""
    gr.Markdown(
        "# Travel Recommendations\n"
        "Browse images, their mean token scores, and VLM-generated US travel "
        "recommendations with extracted cities and states. "
        "Filter by model and political label."
    )

    model_dropdown = gr.Dropdown(
        TRAVEL_MODELS, value=TRAVEL_MODELS[0], label="Model",
    )
    label_dropdown = gr.Dropdown(
        TRAVEL_LABELS, value="ALL", label="Political Label Filter",
    )
    record_dropdown = gr.Dropdown(
        choices=[], label="Select Record", interactive=True,
    )

    with gr.Row():
        with gr.Column(scale=1):
            travel_image = gr.Image(type="pil", label="Portrait")
        with gr.Column(scale=1):
            travel_text = gr.Markdown("Select a record to view travel recommendations.")

    viewer_inputs = [model_dropdown, label_dropdown, record_dropdown]
    updater_inputs = [model_dropdown, label_dropdown]

    model_dropdown.change(
        fn=travel_update_model_or_label,
        inputs=updater_inputs,
        outputs=[record_dropdown, travel_image, travel_text],
    )
    label_dropdown.change(
        fn=travel_update_model_or_label,
        inputs=updater_inputs,
        outputs=[record_dropdown, travel_image, travel_text],
    )
    record_dropdown.change(
        fn=travel_show_record,
        inputs=viewer_inputs,
        outputs=[travel_image, travel_text],
    )
