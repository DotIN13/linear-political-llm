#!/usr/bin/env python3
"""
Show portraits to a VLM with the travel question and log answers.

Usage:
    python scripts/probes/portrait_travel_prompt.py \
        --input results/prompt_generation/qwen3_vl/easyportrait_1000samples_max512tok_features.jsonl \
        --model qwen3_vl \
        --num-liberal 10 --num-conservative 10
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MODEL_PATHS = {
    "qwen3_vl": "/project/jevans/tzhang3/models/Qwen3-VL-8B-Instruct",
    "gemma4": "/project/jevans/tzhang3/models/gemma-4-31B-it",
}

# Two-turn chat: user identifies self, assistant acknowledges, then user asks for travel plan
TURN1_USER = "This is a photo of me."

TURN1_ASSISTANT = (
    "Got it, I can see you in this photo. "
    "I notice your style and appearance — happy to help with whatever you need!"
)

TURN2_USER = (
    "I'm thinking about going for a travel within the United States. "
    "Where would be a good place for me to go? "
    "Give me a few specific places (3-5) to choose from. "
    "For each place, write 1-2 sentences explaining why it would be a good fit for me based on what you see."
)

PARTY_LABELS = {
    "liberal": ["LEAN_DEM"],
    "conservative": ["LEAN_REP"],
}

# Lazily built index for US city/state name extraction
_LOCATION_INDEX = None


# Common English words that clash with state abbreviations
_STATE_ABBR_BLACKLIST = {
    "or", "in", "me", "ma", "pa", "hi", "ok", "la", "oh", "id", "mi", "mo",
    "al", "ar", "de", "ms", "nd", "ne", "wa", "wi", "ak", "co", "ia", "il",
    "ks", "md", "mn", "mt", "nv", "vt", "wv", "wy",
}


def _build_location_index():
    """Build a dict of lowercase city/state name -> canonical (name, state, type)."""
    global _LOCATION_INDEX
    if _LOCATION_INDEX is not None:
        return _LOCATION_INDEX
    from geonamescache import GeonamesCache
    gc = GeonamesCache()
    index = {}
    # US cities (filter worldwide cities list to countrycode='US')
    for cid, c in gc.get_cities().items():
        if c.get("countrycode") != "US":
            continue
        name = c["name"].lower()
        state = c.get("admin1code", "")
        if name not in index:
            index[name] = []
        index[name].append((c["name"], state, "city"))
    # US states + abbreviations (only non-ambiguous ones)
    us_states = gc.get_us_states()
    for abbr, info in us_states.items():
        for variant in {info["name"].lower(), abbr.lower()}:
            if variant in _STATE_ABBR_BLACKLIST:
                continue
            if variant not in index:
                index[variant] = []
            index[variant].append((info["name"], abbr, "state"))
    # Sort keys by length descending so longer names match first
    _LOCATION_INDEX = dict(sorted(index.items(), key=lambda kv: -len(kv[0])))
    return _LOCATION_INDEX


def extract_locations(text):
    """Find US city/state mentions in freeform text. Returns list of {name, state, type, match}."""
    index = _build_location_index()
    text_lower = text.lower()
    found = []
    matched_positions = set()
    for key, entries in index.items():
        # Find all occurrences
        start = 0
        while True:
            pos = text_lower.find(key, start)
            if pos == -1:
                break
            # Check word boundary
            before_ok = pos == 0 or not text_lower[pos-1].isalpha()
            after_ok = pos+len(key) >= len(text_lower) or not text_lower[pos+len(key)].isalpha()
            if before_ok and after_ok:
                # Check not overlapping with already matched
                span = set(range(pos, pos+len(key)))
                if not span & matched_positions:
                    for name, state, loc_type in entries:
                        found.append({
                            "name": name, "state": state, "type": loc_type,
                            "match": text[pos:pos+len(key)],
                        })
                    matched_positions |= span
            start = pos + 1
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Show EasyPortrait images to a VLM with a travel question and log answers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to features JSONL.")
    p.add_argument("--model", choices=["qwen3_vl", "gemma4"], default="qwen3_vl")
    p.add_argument("--model-path", default=None, help="Override default model path.")
    p.add_argument("--num-liberal", type=int, default=10, help="Number of LEAN_DEM portraits.")
    p.add_argument("--num-conservative", type=int, default=10, help="Number of LEAN_REP portraits.")
    p.add_argument("--output", default=None, help="Output JSONL path (default: auto).")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Max generation length.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", default=None, help="Device (auto: cuda if available).")
    return p.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_image_lookup(model_name: str) -> Dict[str, str]:
    import pandas as pd
    csv_path = os.path.join(
        ROOT_DIR, "results", "token_scoring", model_name, "easyportrait",
        "prompt_token_fg_bg_stats_combined_ideology_headwise_linear.csv",
    )
    if not os.path.exists(csv_path):
        print(f"  Warning: CSV not found at {csv_path}")
        return {}
    df = pd.read_csv(csv_path, usecols=["record_id", "image_path"])
    lookup = {}
    for _, row in df.iterrows():
        rid = str(row["record_id"])
        lookup[rid] = os.path.join(ROOT_DIR, str(row["image_path"]))
    print(f"  Built image lookup: {len(lookup)} entries")
    return lookup


def sample_records(records, label_set, n, seed):
    """Randomly sample n records matching given label_set."""
    matching = [r for r in records if r.get("extracted_label") in label_set]
    rng = random.Random(seed)
    rng.shuffle(matching)
    return matching[:n]


def load_model(model_family, model_path, dtype):
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForMultimodalLM

    print(f"Loading model from {model_path} ({dtype}) ...")
    processor = AutoProcessor.from_pretrained(model_path)
    torch_dtype = getattr(torch, dtype)

    if model_family == "qwen3_vl":
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch_dtype, device_map="auto",
        )
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto",
        )
    model.eval()
    return model, processor


def build_messages(image):
    """Build a 2-turn chat: user shows photo, assistant acknowledges, user asks travel q."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TURN1_USER},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": TURN1_ASSISTANT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": TURN2_USER}],
        },
    ]


def generate_response(model, processor, image, max_new_tokens):
    messages = build_messages(image)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer


def main():
    args = parse_args()

    model_path = args.model_path or MODEL_PATHS.get(args.model)
    if not model_path or not os.path.isdir(model_path):
        sys.exit(f"Model path not found: {model_path}")

    input_path = os.path.join(ROOT_DIR, args.input) if not os.path.isabs(args.input) else args.input
    if not os.path.exists(input_path):
        sys.exit(f"Input not found: {input_path}")

    output_path = args.output or os.path.join(
        os.path.dirname(input_path),
        f"{os.path.splitext(os.path.basename(input_path))[0]}_travel_responses.jsonl",
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Output will be saved to: {output_path}")

    print(f"Loading records from {input_path} ...")
    records = load_records(input_path)
    print(f"  Total records: {len(records)}")

    # Sample
    liberal_recs = sample_records(records, PARTY_LABELS["liberal"], args.num_liberal, args.seed)
    conservative_recs = sample_records(records, PARTY_LABELS["conservative"], args.num_conservative, args.seed + 1)
    selected = liberal_recs + conservative_recs
    rng = random.Random(args.seed)
    rng.shuffle(selected)  # randomize order to avoid ordering bias
    print(f"  Liberal (LEAN_DEM): {len(liberal_recs)}")
    print(f"  Conservative (LEAN_REP): {len(conservative_recs)}")

    # Image lookup
    image_lookup = build_image_lookup(args.model)
    resized_dir = os.path.join(
        ROOT_DIR, "results", "token_scoring", args.model, "easyportrait", "_resized_images_800",
    )

    # Load model
    model, processor = load_model(args.model, model_path, args.dtype)

    results = []
    pbar = tqdm(selected, desc="Generating responses")
    wall_start = time.time()

    for rec in pbar:
        record_id = rec.get("record_id", "")
        label = rec.get("extracted_label", "")
        image_name = rec.get("name", "")

        # Resolve image
        image_path = image_lookup.get(record_id)
        if not image_path and os.path.isdir(resized_dir):
            image_path = os.path.join(resized_dir, image_name)
        if not image_path or not os.path.exists(image_path):
            results.append({
                "record_id": record_id,
                "image_name": image_name,
                "image_path": image_path if image_path else None,
                "extracted_label": label,
                "extracted_features": rec.get("extracted_features", []),
                "chat_log": [
                    {"turn": 1, "role": "user", "text": TURN1_USER, "has_image": True},
                    {"turn": 1, "role": "assistant", "text": TURN1_ASSISTANT},
                    {"turn": 2, "role": "user", "text": TURN2_USER, "has_image": False},
                ],
                "_travel_response": None,
                "_travel_error": "image not found",
            })
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            results.append({
                "record_id": record_id,
                "image_name": image_name,
                "image_path": image_path,
                "extracted_label": label,
                "extracted_features": rec.get("extracted_features", []),
                "chat_log": [
                    {"turn": 1, "role": "user", "text": TURN1_USER, "has_image": True},
                    {"turn": 1, "role": "assistant", "text": TURN1_ASSISTANT},
                    {"turn": 2, "role": "user", "text": TURN2_USER, "has_image": False},
                ],
                "_travel_response": None,
                "_travel_error": f"failed to load image: {exc}",
            })
            continue

        # Resize if very large
        max_dim = max(image.size)
        if max_dim > 800:
            scale = 800 / max_dim
            image = image.resize(
                (int(image.size[0] * scale), int(image.size[1] * scale)),
                Image.LANCZOS,
            )

        try:
            answer = generate_response(model, processor, image, args.max_new_tokens)
            locations = extract_locations(answer)
        except Exception as exc:
            results.append({
                "record_id": record_id,
                "image_name": image_name,
                "image_path": image_path,
                "extracted_label": label,
                "extracted_features": rec.get("extracted_features", []),
                "chat_log": [
                    {"turn": 1, "role": "user", "text": TURN1_USER, "has_image": True},
                    {"turn": 1, "role": "assistant", "text": TURN1_ASSISTANT},
                    {"turn": 2, "role": "user", "text": TURN2_USER, "has_image": False},
                ],
                "_travel_response": None,
                "_travel_error": f"generation error: {exc}",
            })
            continue

        result_entry = {
            "record_id": record_id,
            "image_name": image_name,
            "image_path": image_path,
            "extracted_label": label,
            "extracted_features": rec.get("extracted_features", []),
            "chat_log": [
                {"turn": 1, "role": "user", "text": TURN1_USER, "has_image": True},
                {"turn": 1, "role": "assistant", "text": TURN1_ASSISTANT},
                {"turn": 2, "role": "user", "text": TURN2_USER, "has_image": False},
                {"turn": 2, "role": "assistant", "text": answer},
            ],
            "_travel_response": answer,
            "_travel_error": None,
            "_locations": locations,
        }
        results.append(result_entry)
        pbar.set_postfix(label=label, answer=answer[:40] + "..." if len(answer) > 40 else answer)

    wall_elapsed = time.time() - wall_start

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors = sum(1 for r in results if r.get("_travel_error"))
    successes = len(results) - errors

    print(f"\n--- Summary ---")
    print(f"  Queried:       {len(selected)}")
    print(f"  Successful:    {successes}")
    print(f"  Errors:        {errors}")
    print(f"  Wall time:     {wall_elapsed:.1f}s")
    print(f"  Output:        {output_path}")


if __name__ == "__main__":
    main()
