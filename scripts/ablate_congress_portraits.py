"""Ablation study: generate 32 synthetic base portraits + 4 cumulative edit steps each.

Base traits: male, white, blonde, American flag background.
Edit chain (cumulative — each step applied to the previous result):
  1. Remove American flag
  2. Change blonde hair → dark brown
  3. Change skin tone → medium-brown
  4. Change gender → female

Output: data/ablation_study/person_01/step0_base.jpg ... step4_female.jpg
"""

import argparse
import io
import json
import os
import time
import sys
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

# ── 32 varied persona descriptions (same base traits, varied age/appearance) ──

PERSONAS = [
    {"age": 35, "hair": "short neat blonde", "expression": "confident slight smile", "suit": "navy suit with a red tie", "extra": "clean-shaven"},
    {"age": 42, "hair": "short styled blonde", "expression": "warm friendly smile", "suit": "charcoal suit with a blue tie", "extra": "slight stubble"},
    {"age": 50, "hair": "side-parted blonde", "expression": "calm neutral expression", "suit": "dark gray suit with a striped tie", "extra": "wire-rimmed glasses"},
    {"age": 38, "hair": "combed-back blonde", "expression": "professional slight smile", "suit": "black suit with a silver tie", "extra": "clean-shaven jaw"},
    {"age": 55, "hair": "wavy blonde with gray temples", "expression": "authoritative slight smile", "suit": "navy suit with a burgundy tie", "extra": "distinguished look"},
    {"age": 45, "hair": "textured blonde crop", "expression": "friendly grin", "suit": "medium gray suit no tie", "extra": "open collar"},
    {"age": 33, "hair": "short clean blonde cut", "expression": "serious focused look", "suit": "dark blue suit with a gold tie", "extra": "sharp jawline"},
    {"age": 48, "hair": "swept-back blonde", "expression": "pleasant engaging smile", "suit": "charcoal suit with a teal tie", "extra": "slight crow's feet"},
    {"age": 58, "hair": "full blonde with silver streaks", "expression": "wise knowing smile", "suit": "brown tweed jacket", "extra": "distinguished"},
    {"age": 40, "hair": "short spiky blonde", "expression": "energetic smile", "suit": "navy blazer with a light blue tie", "extra": "athletic build"},
    {"age": 52, "hair": "classic side-part blonde", "expression": "reassuring slight smile", "suit": "dark gray suit with a red striped tie", "extra": "mature features"},
    {"age": 37, "hair": "textured quiff blonde", "expression": "confident half-smile", "suit": "black suit with a white shirt no tie", "extra": "modern look"},
    {"age": 60, "hair": "full blonde going gray", "expression": "grandfatherly warm smile", "suit": "navy blazer with a pocket square", "extra": "reading glasses"},
    {"age": 44, "hair": "slicked-back blonde", "expression": "serious professional look", "suit": "charcoal pinstripe suit with a red tie", "extra": "CEO presence"},
    {"age": 31, "hair": "fresh blonde undercut", "expression": "bright enthusiastic smile", "suit": "light gray suit with a navy tie", "extra": "youthful energy"},
    {"age": 56, "hair": "thick blonde swept to side", "expression": "calm pleasant expression", "suit": "dark blue suit with a patterned tie", "extra": "senior authority"},
    {"age": 46, "hair": "neat executive blonde", "expression": "measured slight smile", "suit": "black suit with a silver striped tie", "extra": "faint laugh lines"},
    {"age": 39, "hair": "casual tousled blonde", "expression": "relaxed genuine smile", "suit": "navy sport coat no tie", "extra": "approachable look"},
    {"age": 53, "hair": "full-bodied blonde", "expression": "firm but friendly", "suit": "dark charcoal suit with a burgundy tie", "extra": "strong jaw"},
    {"age": 36, "hair": "short textured blonde fade", "expression": "direct eye contact slight smile", "suit": "medium blue suit with a navy tie", "extra": "modern politician"},
    {"age": 49, "hair": "classic blonde part left", "expression": "contemplative slight smile", "suit": "navy blazer with a red tie", "extra": "horn-rimmed glasses"},
    {"age": 43, "hair": "voluminous blonde sweep", "expression": "confident smirk", "suit": "black turtleneck under a gray blazer", "extra": "artsy intellectual"},
    {"age": 57, "hair": "silver-blonde thinning", "expression": "patient knowing smile", "suit": "dark brown suit with a gold tie", "extra": "seasoned professional"},
    {"age": 34, "hair": "fresh blonde crew cut", "expression": "bright engaged smile", "suit": "navy suit with a light blue tie", "extra": "fit and energetic"},
    {"age": 51, "hair": "side-swept graying blonde", "expression": "thoughtful slight smile", "suit": "charcoal suit with a patterned navy tie", "extra": "policy wonk look"},
    {"age": 47, "hair": "full blonde comb-over", "expression": "confident reassuring smile", "suit": "dark blue suit with a white shirt", "extra": "polished appearance"},
    {"age": 41, "hair": "short messy blonde", "expression": "warm approachable grin", "suit": "gray blazer over a white button-down", "extra": "relatable everyman"},
    {"age": 59, "hair": "distinguished gray-blonde", "expression": "steady composed look", "suit": "navy suit with a red patterned tie", "extra": "elder statesman"},
    {"age": 32, "hair": "fresh blonde taper cut", "expression": "optimistic bright smile", "suit": "light navy suit with a coral tie", "extra": "young reformer"},
    {"age": 54, "hair": "mature blonde full", "expression": "composed slight smile", "suit": "charcoal suit with a steel blue tie", "extra": "seasoned authority"},
    {"age": 38, "hair": "dynamic blonde with lift", "expression": "engaging warm smile", "suit": "dark gray suit with a emerald tie", "extra": "charismatic speaker"},
    {"age": 61, "hair": "dignified gray-blonde", "expression": "warm but serious", "suit": "navy blazer with a red bow tie", "extra": "old-school charm"},
]


def _extract_first_image(response) -> Image.Image:
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and getattr(inline_data, "data", None):
                return Image.open(io.BytesIO(inline_data.data)).convert("RGB")
    raise RuntimeError("No image found in Gemini response.")


def _to_image_part(image: Image.Image) -> types.Part:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")


def generate_base(client: genai.Client, model: str, persona: dict) -> Image.Image:
    prompt = (
        f"Photorealistic editorial portrait of a {persona['age']}-year-old white male "
        f"with {persona['hair']} hair, wearing a {persona['suit']}. "
        f"He has a {persona['expression']}, {persona['extra']}. "
        f"He is standing in front of an American flag hanging on a wall behind him. "
        f"Professional head-and-shoulders studio shot, soft even lighting, plain neutral backdrop around the flag. "
        f"No text, no logos, no watermarks."
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    return _extract_first_image(response)


def edit_image(client: genai.Client, model: str, image: Image.Image, instruction: str) -> Image.Image:
    response = client.models.generate_content(
        model=model,
        contents=[_to_image_part(image), instruction],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )
    return _extract_first_image(response)


EDIT_INSTRUCTIONS = [
    (
        "step1_no_flag",
        "Remove the American flag from the background completely. "
        "Replace the flag with a plain neutral gray office wall that matches the surrounding area. "
        "Do NOT change the person's face, hair color, hair style, skin tone, clothing, pose, expression, "
        "or lighting in any way. Only change the background area where the flag was.",
    ),
    (
        "step2_dark_hair",
        "Change the person's hair color from blonde to dark brown. "
        "Keep the exact same hairstyle, volume, and texture. "
        "Do NOT change anything else — face, skin tone, clothing, background, pose, expression, "
        "and lighting must remain absolutely identical.",
    ),
    (
        "step3_nonwhite",
        "Change the person's skin tone to a medium-brown complexion. "
        "Keep the exact same facial structure, features, and identity. "
        "Do NOT change anything else — hair color, hair style, clothing, background, pose, expression, "
        "and lighting must remain absolutely identical.",
    ),
    (
        "step4_female",
        "Change this person to a female version while preserving the exact same age, pose, expression, "
        "outfit, hair color, hair style, skin tone, and background. "
        "The person should have feminine facial features (softer jawline, slightly fuller lips, "
        "more delicate bone structure) but remain recognizably based on the same person. "
        "Do NOT change anything else — clothing, background, lighting, pose must remain identical.",
    ),
]


def process_one_person(client, model, persona, person_idx, output_dir, max_retries=3):
    person_dir = output_dir / f"person_{person_idx:02d}"
    person_dir.mkdir(parents=True, exist_ok=True)

    # Check if this person is already done
    expected_files = ["step0_base.jpg"] + [f"{name}.jpg" for name, _ in EDIT_INSTRUCTIONS]
    if all((person_dir / f).exists() for f in expected_files):
        print(f"  Person {person_idx:02d}: already complete, skipping")
        return True

    # Save persona metadata
    with open(person_dir / "persona.json", "w") as f:
        json.dump(persona, f, indent=2)

    # ── Step 0: Generate base portrait ────────────────────────────────────
    for attempt in range(max_retries):
        try:
            print(f"  Person {person_idx:02d}: generating base ...")
            current = generate_base(client, model, persona)
            current.save(person_dir / "step0_base.jpg", quality=95)
            print(f"    step0_base.jpg saved ({current.size[0]}x{current.size[1]})")
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"    Base generation attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"    Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"    FAILED base generation for person {person_idx:02d}")
                return False

    # ── Steps 1-4: Cumulative edits ───────────────────────────────────────
    for step_name, instruction in EDIT_INSTRUCTIONS:
        output_path = person_dir / f"{step_name}.jpg"
        if output_path.exists():
            print(f"    {step_name}.jpg already exists, using as current")
            current = Image.open(output_path)
            continue

        for attempt in range(max_retries):
            try:
                print(f"    Editing → {step_name} ...")
                edited = edit_image(client, model, current, instruction)
                edited.save(output_path, quality=95)
                print(f"    {step_name}.jpg saved ({edited.size[0]}x{edited.size[1]})")
                current = edited
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    Edit attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"    Retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"    FAILED edit {step_name} for person {person_idx:02d}")
                    return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study portraits")
    parser.add_argument("--output-dir", default="data/ablation_study")
    parser.add_argument("--model", default="gemini-3.1-flash-image-preview")
    parser.add_argument("--num-persons", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client()

    personas = PERSONAS[:args.num_persons]
    total = len(personas)
    success = 0
    failed = 0

    for i in range(args.start_index, total):
        persona = personas[i]
        print(f"\n{'='*60}")
        print(f"Person {i+1:02d}/{total:02d}: age={persona['age']}, {persona['suit']}")
        print(f"{'='*60}")

        ok = process_one_person(client, args.model, persona, i + 1, output_dir, args.max_retries)
        if ok:
            success += 1
        else:
            failed += 1

        time.sleep(1)  # rate limit safety

    print(f"\n{'='*60}")
    print(f"DONE. {success} succeeded, {failed} failed out of {total}")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
