#!/usr/bin/env python3
"""
Generate score-data JSONL for paired Gemini image datasets.
"""

import argparse
import glob
import json
import os
from typing import List

from PIL import Image


DEFAULT_IMAGE_DIR = "data/gemini_tie_pairs/red_blue"
DEFAULT_PROMPT = (
    "write a possible news article that would best accompany this image, from the best "
    "possible media outlet. Use your knowledge about American political culture and the "
    "media landscape in doing so. If the image seems slanted, write a slanted article. "
    "Do not worry about neutrality: write the article in the style of the media outlet "
    "most likely to have used this image."
)
DEFAULT_OUTPUT_PATH = "data/probes/red_blue_score_data.jsonl"
DEFAULT_ID_PREFIX = "red_blue"
DEFAULT_SOURCE = "gemini_tie_pairs_red_blue"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build red-blue score-data JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--id-prefix", default=DEFAULT_ID_PREFIX)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    return parser.parse_args()


def collect_image_paths(image_dir: str, recursive: bool, limit: int = None) -> List[str]:
    image_paths: List[str] = []
    for pattern in IMAGE_EXTENSIONS:
        query = os.path.join(image_dir, "**", pattern) if recursive else os.path.join(image_dir, pattern)
        image_paths.extend(glob.glob(query, recursive=recursive))

    image_paths = sorted(set(image_paths))
    if not image_paths and not recursive:
        for pattern in IMAGE_EXTENSIONS:
            query = os.path.join(image_dir, "**", pattern)
            image_paths.extend(glob.glob(query, recursive=True))
        image_paths = sorted(set(image_paths))
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {image_dir}. If your files are in nested folders, use --recursive."
        )
    return image_paths


def dims(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    return int(width), int(height)


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.image_dir, recursive=args.recursive, limit=args.limit)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as handle:
        for idx, image_path in enumerate(image_paths):
            width, height = dims(image_path)
            payload = {
                "id": f"{args.id_prefix}_{idx:06d}",
                "source": args.source,
                "name": os.path.basename(image_path),
                "image_path": image_path,
                "prompt": args.prompt,
                "text": args.prompt,
                "image_width": width,
                "image_height": height,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.prompt},
                            {"type": "image", "image": image_path},
                        ],
                    }
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Wrote {len(image_paths)} rows: {args.output_path}")


if __name__ == "__main__":
    main()
