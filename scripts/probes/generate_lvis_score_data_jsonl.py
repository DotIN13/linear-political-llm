#!/usr/bin/env python3
"""
Generate score-data JSONL for LVIS training images.

Reads LVIS v1 train annotations and enumerates all unique images,
writing a JSONL manifest compatible with scripts/probes/token_scoring.py.

Image paths point to datasets/lvis/train/<coco_file_name>.
Images are resized to <=800px by token_scoring.py at scoring time.
"""

import argparse
import json
import os
from typing import Dict, List, Set


DEFAULT_LVIS_JSON = "datasets/lvis/lvis_v1_train.json"
DEFAULT_IMAGE_ROOT = "datasets/lvis/train"
DEFAULT_PROMPT = (
    "write a possible news article that would best accompany this image, from the best "
    "possible media outlet. Use your knowledge about American political culture and the "
    "media landscape in doing so. If the image seems slanted, write a slanted article. "
    "Do not worry about neutrality: write the article in the style of the media outlet "
    "most likely to have used this image."
)
DEFAULT_OUTPUT_PATH = "data/probes/lvis_score_data.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LVIS score-data JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lvis-json", default=DEFAULT_LVIS_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def build_image_index(images: List[Dict]) -> Dict[int, Dict]:
    return {img["id"]: img for img in images}


def extract_file_name(coco_url: str) -> str:
    return coco_url.rsplit("/", 2)[-2] + "/" + coco_url.rsplit("/", 1)[-1]


def main() -> None:
    args = parse_args()

    with open(args.lvis_json, encoding="utf-8") as handle:
        lvis = json.load(handle)

    image_index = build_image_index(lvis["images"])

    unique_image_ids: Set[int] = set()
    for ann in lvis["annotations"]:
        unique_image_ids.add(int(ann["image_id"]))

    sorted_ids = sorted(unique_image_ids)
    if args.limit is not None:
        sorted_ids = sorted_ids[: args.limit]

    missing = 0
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as handle:
        for idx, image_id in enumerate(sorted_ids):
            img = image_index.get(image_id)
            if img is None:
                missing += 1
                continue

            file_name = extract_file_name(img["coco_url"])
            image_path = os.path.join(args.image_root, file_name)

            if not os.path.exists(image_path):
                missing += 1
                continue

            payload = {
                "id": f"lvis_{image_id:06d}",
                "source": "lvis",
                "name": file_name,
                "image_path": image_path,
                "prompt": args.prompt,
                "text": args.prompt,
                "image_width": int(img["width"]),
                "image_height": int(img["height"]),
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

    print(f"Wrote {idx + 1 - missing} rows (skipped {missing} missing images): {args.output_path}")


if __name__ == "__main__":
    main()
