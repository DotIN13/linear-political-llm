"""One-off: do the model actually read the images?

Builds *photos* trials through the same modules the v1 pilot uses
(``read_items`` -> ``build_base`` -> the ``vllm`` adaptor), but replaces the
digest ask with "describe these images". The item's LVIS categories (recorded by
the sampler) are printed next to the reply, so a match is evidence the image
tokens reached the model rather than being dropped.

A ``no_photos`` trial of the same question is the control: with no image parts
the model should say it cannot see anything.

    python -m bench_v2.oneoffs.describe_images
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_v2 import registry
from bench_v2.helpers.dataset import read_items
from bench_v2.helpers.generation import build_base
from bench_v2.paths import items_dir

ITEMS = Path(items_dir()) / "explore_extreme_v1.jsonl"
BUCKETS = ("low", "mid", "high")
QUESTION = ("Describe what is in each of the images above, one sentence per image. "
            "Be concrete about the objects you can see.")
MAX_NEW_TOKENS = 350


def one(adaptor, item, condition, label):
    trial = build_base(
        surface="describe_images", item=item, condition=condition,
        variant={"scheme": "chat", "question": "q0", "clause": "bare"},
        seed=42,
        question_fn=lambda qid, order, attribution: QUESTION,
        max_new_tokens=MAX_NEW_TOKENS,
        randomizes_per_item=False,
    )
    resp = adaptor.run(trial)
    usage = resp.usage or {}
    print(f"\n===== {label}  item={item.item_id}  condition={condition} =====")
    print(f"images: {item.images}")
    print(f"known LVIS categories: {item.covariates.get('categories')}")
    print(f"n_images_sent={len(trial.conversation.images)}  "
          f"prompt_tokens={usage.get('prompt_tokens')}  error={resp.error}")
    print("--- model: ---")
    print((resp.text or "").strip() or "(empty)")
    return resp


def main() -> int:
    registry.load_adaptors()
    adaptor = registry.get_adaptor("vllm")(seed=42)
    adaptor.setup()

    items, synthetic = read_items(ITEMS, limit=0)
    if synthetic:
        print(f"[describe] no items at {ITEMS}; using synthetic no-image items")

    picked = {}
    for item in items:
        bucket = item.covariates.get("bucket")
        if bucket in BUCKETS and bucket not in picked:
            picked[bucket] = item
    chosen = [picked[b] for b in BUCKETS if b in picked] or items[:1]

    # Control first: same question, no images.
    one(adaptor, chosen[0], "no_photos", "CONTROL no images")
    for item in chosen:
        one(adaptor, item, "photos", f"PHOTOS bucket={item.covariates.get('bucket')}")

    adaptor.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
