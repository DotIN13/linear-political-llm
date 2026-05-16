"""Label congress portrait images with Gemini.

Outputs a CSV with columns:
    bioguide_id, speculated_age, gender, is_white, is_blonde,
    is_smiling, has_american_flag, wears_dark_suit

Resumes from existing output if present.
"""

import argparse
import csv
import json
import os
import re
import time
import sys
from pathlib import Path

from google import genai
from google.genai import types

IMAGES_DIR = Path("data/congress_images")
OUTPUT_CSV = Path("data/congress_portrait_labels.csv")

PROMPT = """Analyze this congress portrait photo and return a JSON object with exactly these fields (no markdown, no extra text):

{
  "speculated_age": <integer age estimate>,
  "gender": "<male or female>",
  "is_white": <true or false>,
  "is_blonde": <true or false>,
  "is_smiling": <true or false>,
  "has_american_flag": <true or false>,
  "wears_dark_suit": <true or false>
}

is_blonde means the person has blonde/light-colored hair of any shade (not brown, black, gray, or white). Only return true if hair is clearly visible and blonde.
has_american_flag means there is a visible American flag anywhere in the background or on clothing.
wears_dark_suit means the person is wearing a dark-colored suit jacket (navy, charcoal, black, dark blue).

Return ONLY the JSON object, nothing else."""


def load_existing(output_path):
    existing = {}
    if output_path.exists():
        with open(output_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["bioguide_id"]] = row
        print(f"Loaded {len(existing)} existing labels from {output_path}")
    return existing


def save_results(results, output_path, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bioguide_id in sorted(results.keys()):
            writer.writerow(results[bioguide_id])


def get_image_files(images_dir):
    return sorted(images_dir.glob("*.jpg"))


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def parse_labels(data):
    return {
        "speculated_age": int(data["speculated_age"]),
        "gender": str(data["gender"]).strip().lower(),
        "is_white": bool(data["is_white"]),
        "is_blonde": bool(data["is_blonde"]),
        "is_smiling": bool(data["is_smiling"]),
        "has_american_flag": bool(data["has_american_flag"]),
        "wears_dark_suit": bool(data["wears_dark_suit"]),
    }


def label_image(client, model, image_path, max_retries=3):
    bioguide_id = image_path.stem

    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            response = client.models.generate_content(
                model=model,
                contents=[image_part, PROMPT],
            )

            result = parse_labels(extract_json(response.text))
            result["bioguide_id"] = bioguide_id
            return result

        except Exception as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt + 1}/{max_retries} failed for {bioguide_id}: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  All retries exhausted for {bioguide_id}")
                raise


def main():
    parser = argparse.ArgumentParser(description="Label congress portraits with Gemini")
    parser.add_argument("--images-dir", default=str(IMAGES_DIR))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save CSV every N images")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start processing at this index (0-based)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)

    image_files = get_image_files(images_dir)
    print(f"Found {len(image_files)} images in {images_dir}")

    results = load_existing(output_path)

    fieldnames = [
        "bioguide_id", "speculated_age", "gender", "is_white",
        "is_blonde", "is_smiling", "has_american_flag", "wears_dark_suit",
    ]

    client = genai.Client()

    total = len(image_files)
    processed = 0
    failed = 0

    for i in range(args.start_index, total):
        image_path = image_files[i]
        bioguide_id = image_path.stem

        if bioguide_id in results:
            continue

        print(f"[{i + 1}/{total}] Labeling {bioguide_id} ...", end=" ", flush=True)

        try:
            result = label_image(client, args.model, image_path)
            results[bioguide_id] = result
            print(f"age={result['speculated_age']} gender={result['gender']} "
                  f"white={result['is_white']} blonde={result['is_blonde']} "
                  f"smile={result['is_smiling']} flag={result['has_american_flag']} "
                  f"suit={result['wears_dark_suit']}")
            processed += 1

        except Exception:
            failed += 1
            failed_row = {
                "bioguide_id": bioguide_id,
                "speculated_age": "",
                "gender": "",
                "is_white": "",
                "is_blonde": "",
                "is_smiling": "",
                "has_american_flag": "",
                "wears_dark_suit": "",
            }
            results[bioguide_id] = failed_row
            print("FAILED")

        if (processed + failed) % args.save_interval == 0 or i == total - 1:
            save_results(results, output_path, fieldnames)
            print(f"  --- Saved {len(results)} entries to {output_path} ---")

        time.sleep(0.5)

    save_results(results, output_path, fieldnames)

    print(f"\nDone. {processed} labeled, {failed} failed, {len(results)} total in {output_path}")


if __name__ == "__main__":
    main()