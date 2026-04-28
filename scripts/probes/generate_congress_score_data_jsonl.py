#!/usr/bin/env python3
"""
Generate score-data JSONL for Congress portraits.

This follows the data selection conventions from:
  scripts/run_qwen3_vl_congress_ideology_tokens.py
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from PIL import Image, ImageOps


DEFAULT_IMAGE_DIR = "data/congress_images"
DEFAULT_HS_PATH = "data/HS116_members.csv"
DEFAULT_CUR_PATH = "data/legislators-current.json"
DEFAULT_HIST_PATH = "data/legislators-historical.json"
DEFAULT_PROMPT = "What's his/her position on US politics?"
DEFAULT_RESIZED_IMAGE_WIDTH = 250
DEFAULT_OUTPUT_PATH = "data/probes/congress_score_data.jsonl"

PARTY_MAP = {100: "Democrat", 200: "Republican"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Congress score-data JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--hs-path", default=DEFAULT_HS_PATH)
    parser.add_argument("--current-legislators-path", default=DEFAULT_CUR_PATH)
    parser.add_argument("--historical-legislators-path", default=DEFAULT_HIST_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resized-image-width", type=int, default=DEFAULT_RESIZED_IMAGE_WIDTH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


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
    return df_hs[df_hs["has_image"]].copy().reset_index(drop=True)


def compute_resized_dims(image_path: str, target_width: int) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width > target_width:
            target_height = max(1, round(image.height * target_width / image.width))
            return int(target_width), int(target_height)
        return int(image.width), int(image.height)


def main() -> None:
    args = parse_args()
    df_probe = build_probe_dataframe(
        image_dir=args.image_dir,
        hs_path=args.hs_path,
        current_legislators_path=args.current_legislators_path,
        historical_legislators_path=args.historical_legislators_path,
    )

    rows = list(df_probe.itertuples(index=False))
    if args.limit is not None:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    count = 0
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            resized_w, resized_h = compute_resized_dims(row.image_path, args.resized_image_width)
            party_code = int(row.party_code)
            party = PARTY_MAP.get(party_code, f"Other ({party_code})")

            payload = {
                "id": f"congress_{row.bioguide}",
                "source": "congress_images",
                "name": str(row.name),
                "bioguide": str(row.bioguide),
                "party": party,
                "party_code": party_code,
                "label": float(row.nominate_dim1),
                "nominate_dim1": float(row.nominate_dim1),
                "image_path": str(row.image_path),
                "prompt": args.prompt,
                "text": args.prompt,
                "resized_width": resized_w,
                "resized_height": resized_h,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(row.image_path)},
                            {"type": "text", "text": args.prompt},
                        ],
                    }
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows: {args.output_path}")


if __name__ == "__main__":
    main()
