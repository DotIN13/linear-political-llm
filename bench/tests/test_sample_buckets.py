"""Round-4 sampler: three fixed-threshold buckets replace deciles as the default.

The buckets are ``low`` (<-0.5) / ``mid`` ([-0.5,+0.5]) / ``high`` (>+0.5),
the item_id carries the bucket name, ``primary_iv`` is ``bucket`` (ordinal
-1/0/+1), three images still come from one bucket, and the ``n_objects``
(distinct LVIS category count) balance is moved from strata onto buckets.
"""

import pytest

from bench.sample import (
    BUCKETS, BUCKET_ORDINAL, assign_buckets, bucket_of, make_bucket_items,
)

IMAGES_PER_ITEM = 3


def _pool():
    """Synthetic pool spanning all three buckets, with a category-count gradient
    so the balance has something to do."""
    rows = []
    points = [
        ("low", -0.7, -0.55), ("low", -0.55, -0.51),
        ("mid", -0.5, 0.5), ("mid", -0.1, 0.3),
        ("high", 0.51, 0.6), ("high", 0.7, 0.9),
    ]
    i = 0
    for bucket, lo, hi in points:
        for _ in range(60):
            image_mean = lo + (hi - lo) * (_ % 10) / 9.0
            rows.append({
                "record_id": f"lvis_{i:06d}",
                "record_name": f"train2017/{i:012d}.jpg",
                "image_path": f"/tmp/{i}.jpg",
                "image_h": 480, "image_w": 640,
                "num_image_tokens": 400, "image_token_mismatch": 0,
                "image_mean": image_mean,
                "coco_id": i,
                # n_objects is the *instance* count; categories the distinct list
                "n_objects": 3 + (i % 20),
                "categories": [f"cat{j}" for j in range(1 + (i % 9))],
                "n_persons": 0, "has_text_cat": False, "text_cats": [],
                "aspect": 1.3,
            })
            i += 1
    return rows


def test_bucket_of_thresholds():
    assert bucket_of(-0.51) == "low"
    assert bucket_of(-0.5) == "mid"
    assert bucket_of(0.0) == "mid"
    assert bucket_of(0.5) == "mid"
    assert bucket_of(0.51) == "high"


def test_assign_buckets_sets_name_and_ordinal():
    rows = _pool()
    tagged = assign_buckets(rows)
    for row in tagged:
        assert row["bucket"] == bucket_of(row["image_mean"])
        assert row["stratum"] == BUCKET_ORDINAL[row["bucket"]]
    assert set(r["bucket"] for r in tagged) == set(BUCKETS)


def test_three_images_per_item_all_from_one_bucket():
    items, _ = make_bucket_items(_pool(), per_bucket=60, images_per_item=IMAGES_PER_ITEM,
                                 splits=("explore", "confirm"), seed=7)
    for split_items in items.values():
        for item in split_items:
            assert len(item["images"]) == IMAGES_PER_ITEM
            assert len(item["image_scores"]) == IMAGES_PER_ITEM
            # all three image_means fall inside the item's bucket
            assert {bucket_of(s) for s in item["image_scores"]} == {item["bucket"]}


def test_item_id_carries_the_bucket_name():
    items, _ = make_bucket_items(_pool(), per_bucket=60, images_per_item=IMAGES_PER_ITEM,
                                 splits=("explore",), seed=7)
    for item in items["explore"]:
        abbrev = {"low": "lo", "mid": "mid", "high": "hi"}[item["bucket"]]
        assert item["item_id"].startswith(f"lvis3_{abbrev}_")
        assert item["primary_iv"] == "bucket"
        assert item["covariates"]["bucket"] == item["bucket"]


def test_item_ids_unique_and_per_bucket_counter():
    items, _ = make_bucket_items(_pool(), per_bucket=60, images_per_item=IMAGES_PER_ITEM,
                                 splits=("explore", "confirm"), seed=7)
    ids = [i["item_id"] for split in items.values() for i in split]
    assert len(ids) == len(set(ids))
    # every bucket's own counter restarts at 00000
    assert any(i.startswith("lvis3_lo_00000") for i in ids)
    assert any(i.startswith("lvis3_mid_00000") for i in ids)
    assert any(i.startswith("lvis3_hi_00000") for i in ids)


def test_profile_reports_bucket_counts():
    items, profile = _ = make_bucket_items(_pool(), per_bucket=60,
                                           images_per_item=IMAGES_PER_ITEM,
                                           splits=("explore", "confirm"), seed=7)
    assert profile["primary_iv"] == "bucket"
    assert len(profile["buckets"]) == len(BUCKETS)
    for row in profile["buckets"]:
        assert row["n_images_selected"] > 0
        assert "n_categories_mean" in row


def test_raises_when_per_bucket_too_small():
    with pytest.raises(ValueError):
        make_bucket_items(_pool(), per_bucket=3, images_per_item=IMAGES_PER_ITEM,
                          splits=("explore", "confirm"), seed=7)
