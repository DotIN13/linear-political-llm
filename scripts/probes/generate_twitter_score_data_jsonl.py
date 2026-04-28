#!/usr/bin/env python3
"""
Generate score-data JSONL for Twitter images.

This follows image collection and prompt defaults from:
  scripts/run_qwen3_vl_twitter_images.py
"""

import argparse
import glob
import json
import os
from typing import List

from PIL import Image


DEFAULT_IMAGE_DIR = "data/twitter_images"
DEFAULT_MAX_WIDTH = 800
DEFAULT_PROMPT = (
    "Analyze this image. What political perspective or bias might be present? "
    "If this image were shared on Twitter, what political leaning would be most likely to share it? "
    "Consider visual elements, context, and any accompanying sentiment."
)
DEFAULT_OUTPUT_PATH = "data/probes/twitter_images_score_data.jsonl"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build twitter-images score-data JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def collect_image_paths(image_dir: str, recursive: bool, limit: int = None) -> List[str]:
    image_paths: List[str] = []
    for pattern in IMAGE_EXTENSIONS:
        query = os.path.join(image_dir, "**", pattern) if recursive else os.path.join(image_dir, pattern)
        image_paths.extend(glob.glob(query, recursive=recursive))

    image_paths = sorted(set(image_paths))
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return image_paths


def resized_dims(path: str, max_width: int) -> tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    if max_width and max_width > 0 and width > max_width:
        resized_height = max(1, int(round(height * (float(max_width) / float(width)))))
        return int(max_width), int(resized_height)
    return int(width), int(height)


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.image_dir, recursive=args.recursive, limit=args.limit)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as handle:
        for idx, image_path in enumerate(image_paths):
            resized_w, resized_h = resized_dims(image_path, args.max_width)
            payload = {
                "id": f"twitter_{idx:06d}",
                "source": "twitter_images",
                "name": os.path.basename(image_path),
                "image_path": image_path,
                "prompt": args.prompt,
                "text": args.prompt,
                "resized_width": resized_w,
                "resized_height": resized_h,
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
