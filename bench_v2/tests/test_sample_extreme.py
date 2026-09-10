"""Extreme bucketing: label each image by its signed extreme patch score and cut
into equal-count terciles.

The mean-bucket path is unchanged; this checks the new ``bucket_by="extreme"``
path and that the sampled items still load as ``bench_v2.types.Item``.
"""

from __future__ import annotations

import json

from bench_v2.sample import (
    BUCKET_ORDINAL, make_bucket_items, signed_extreme, tercile_cuts,
)
from bench_v2.types import Item

IMAGES_PER_ITEM = 3


def _row(i: int, image_mean: float, image_extreme: float) -> dict:
    return {
        "record_id": f"lvis_{i:06d}",
        "record_name": f"train2017/{i:012d}.jpg",
        "image_path": f"/tmp/{i}.jpg",
        "image_h": 480, "image_w": 640,
        "num_image_tokens": 400, "image_token_mismatch": 0,
        "image_mean": image_mean,
        "image_extreme": image_extreme,
        "coco_id": i,
        "n_objects": 3 + (i % 20),
        "categories": [f"cat{j}" for j in range(1 + (i % 9))],
        "n_persons": 0, "has_text_cat": False, "text_cats": [],
        "aspect": 1.3,
    }


def _pool():
    # extreme runs -1.0..+1.0 in 0.1 steps; mean is deliberately unrelated, so a
    # mean-bucket and an extreme-bucket disagree.
    return [_row(i, image_mean=(-1) ** i * 0.2, image_extreme=-1.0 + 0.1 * i)
            for i in range(21)]


def test_signed_extreme_keeps_the_further_patch():
    assert signed_extreme(-10.0, 5.0) == -10.0
    assert signed_extreme(-3.0, 4.0) == 4.0
    assert signed_extreme(-2.0, 2.0) == -2.0  # tie -> min


def test_tercile_cuts_are_equal_count():
    values = list(range(9))
    lo, hi = tercile_cuts(values)
    assert lo == 3 and hi == 6


def test_extreme_buckets_are_equal_count_and_ordinal():
    items, profile = make_bucket_items(
        _pool(), per_bucket=21, images_per_item=IMAGES_PER_ITEM,
        splits=("explore",), seed=7, bucket_by="extreme")
    assert profile["bucket_by"] == "extreme"
    counts = {row["bucket"]: row["n_images_selected"] for row in profile["buckets"]}
    assert sum(counts.values()) == 21
    assert all(abs(c - 21 / 3) <= 1 for c in counts.values())  # near-equal terciles
    for item in items["explore"]:
        assert item["stratum"] == BUCKET_ORDINAL[item["bucket"]]
        assert len(item["covariates"]["image_extreme"]) == IMAGES_PER_ITEM


def test_three_images_per_item_share_the_extreme_bucket():
    items, _ = make_bucket_items(
        _pool(), per_bucket=21, images_per_item=IMAGES_PER_ITEM,
        splits=("explore",), seed=7, bucket_by="extreme")
    lo, hi = tercile_cuts([r["image_extreme"] for r in _pool()])

    def label(v):
        return "low" if v <= lo else ("mid" if v <= hi else "high")

    for item in items["explore"]:
        assert {label(v) for v in item["covariates"]["image_extreme"]} == {item["bucket"]}


def test_items_load_as_types_item():
    items, _ = make_bucket_items(
        _pool(), per_bucket=21, images_per_item=IMAGES_PER_ITEM,
        splits=("explore",), seed=7, bucket_by="extreme")
    for payload in items["explore"]:
        item = Item.from_dict(payload)
        assert len(item.images) == IMAGES_PER_ITEM
        assert item.covariates["bucket_by"] == "extreme"
        assert item.covariates["bucket"] in ("low", "mid", "high")
        json.dumps(payload)  # round-trips


def test_mean_path_is_unchanged():
    items, profile = make_bucket_items(
        _pool(), per_bucket=21, images_per_item=IMAGES_PER_ITEM,
        splits=("explore",), seed=7)  # default bucket_by="mean"
    assert profile["bucket_by"] == "mean"
    assert profile["bucket_field"] == "image_mean"
