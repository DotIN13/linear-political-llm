"""Sampler: filters, decile strata, 3-images-same-decile, explore/confirm disjoint."""

import random

import pytest

from bench.sample import (
    apply_filters, assign_deciles, coco_id_from_record_name, make_items, parse_filters,
)

BINS = 10
PER_BIN = 30           # 30 images/decile -> 10 items/decile -> 5 explore + 5 confirm
IMAGES_PER_ITEM = 3


def _pool(n_per_bin=60, bins=BINS):
    """Synthetic pool with a clean image_mean gradient, no LVIS file needed."""
    rows, meta = [], {}
    for b in range(bins):
        for i in range(n_per_bin):
            coco_id = b * 1000 + i
            rows.append({
                "record_id": f"lvis_{coco_id:06d}",
                "record_name": f"train2017/{coco_id:012d}.jpg",
                "image_path": f"/tmp/{coco_id}.jpg",
                "image_h": 480, "image_w": 640,
                "num_image_tokens": 400, "image_token_mismatch": 0,
                "image_mean": -0.5 + b * 0.1 + i * 0.0001,
                "coco_id": coco_id,
            })
            meta[coco_id] = {"n_objects": 3 + (i % 13), "categories": ["cup", "table"]}
    return rows, meta


def test_coco_id_parsing():
    assert coco_id_from_record_name("train2017/000000000030.jpg") == 30


def test_parse_filters():
    parsed = parse_filters("no_person,no_text_cats,objects:3-15,aspect:0.6-1.7,tokens:300-800")
    assert parsed["no_person"] and parsed["no_text_cats"]
    assert parsed["objects"] == (3.0, 15.0)
    assert parsed["aspect"] == (0.6, 1.7)
    assert parsed["tokens"] == (300.0, 800.0)
    with pytest.raises(ValueError):
        parse_filters("nonsense")


def test_filters_drop_the_right_rows():
    rows, meta = _pool(n_per_bin=1, bins=1)
    base = rows[0]

    def one(row_over, meta_over):
        row = dict(base); row.update(row_over)
        m = dict(meta[base["coco_id"]]); m.update(meta_over)
        return apply_filters([row], {base["coco_id"]: m}, parse_filters(
            "no_person,no_text_cats,objects:3-15,aspect:0.6-1.7,tokens:300-800"))

    assert len(one({}, {})[0]) == 1
    assert one({}, {"categories": ["person", "cup"]})[1]["F2_person"] == 1
    assert one({}, {"categories": ["signboard"]})[1]["F3_text_category"] == 1
    assert one({}, {"categories": ["flag"]})[1]["F3_text_category"] == 1
    assert one({}, {"n_objects": 20})[1]["F5_n_objects"] == 1
    assert one({}, {"n_objects": 2})[1]["F5_n_objects"] == 1
    assert one({"image_w": 1600, "image_h": 480}, {})[1]["F6_aspect"] == 1
    assert one({"num_image_tokens": 100}, {})[1]["F1_token_count"] == 1
    assert one({"image_token_mismatch": 1}, {})[1]["F1_token_mismatch"] == 1
    assert apply_filters([base], {}, parse_filters("no_person"))[1]["no_lvis_annotation"] == 1


def test_deciles_are_equal_count_and_monotone():
    rows, _ = _pool()
    tagged = assign_deciles(rows, bins=BINS)
    counts = {}
    for row in tagged:
        counts[row["decile"]] = counts.get(row["decile"], 0) + 1
    assert set(counts) == set(range(BINS))
    assert len(set(counts.values())) == 1, counts
    means = {}
    for row in tagged:
        means.setdefault(row["decile"], []).append(row["image_mean"])
    ordered = [sum(v) / len(v) for _, v in sorted(means.items())]
    assert ordered == sorted(ordered)


def _items():
    rows, meta = _pool()
    kept, _ = apply_filters(rows, meta, parse_filters(
        "no_person,no_text_cats,objects:3-15,aspect:0.6-1.7,tokens:300-800"))
    return make_items(kept, bins=BINS, per_bin=PER_BIN,
                      images_per_item=IMAGES_PER_ITEM, splits=("explore", "confirm"), seed=7)


def test_stratum_counts_are_exact():
    items, profile = _items()
    per_split = (PER_BIN // IMAGES_PER_ITEM) // 2
    for split in ("explore", "confirm"):
        assert len(items[split]) == per_split * BINS
        counts = {}
        for item in items[split]:
            counts[item["decile"]] = counts.get(item["decile"], 0) + 1
        assert counts == {d: per_split for d in range(BINS)}
    assert profile["n_items"] == {"explore": per_split * BINS, "confirm": per_split * BINS}


def test_three_images_per_item_all_from_the_same_decile():
    items, _ = _items()
    rows, _meta = _pool()
    decile_of = {r["record_name"]: r["decile"] for r in assign_deciles(rows, bins=BINS)}
    for split_items in items.values():
        for item in split_items:
            assert len(item["images"]) == IMAGES_PER_ITEM
            assert len(item["image_scores"]) == IMAGES_PER_ITEM
            assert {decile_of[name] for name in item["images"]} == {item["decile"]}


def test_explore_and_confirm_share_no_item_and_no_image():
    items, _ = _items()
    ids = {s: {i["item_id"] for i in v} for s, v in items.items()}
    assert ids["explore"] & ids["confirm"] == set()
    imgs = {s: {n for i in v for n in i["images"]} for s, v in items.items()}
    assert imgs["explore"] & imgs["confirm"] == set()
    # and no image is reused inside a split either
    for split, split_items in items.items():
        flat = [n for i in split_items for n in i["images"]]
        assert len(flat) == len(set(flat)), f"{split} reuses an image"


def test_sampling_is_seed_reproducible():
    a, _ = _items()
    b, _ = _items()
    assert [i["images"] for i in a["explore"]] == [i["images"] for i in b["explore"]]


def test_raises_when_per_bin_too_small_to_split():
    rows, meta = _pool()
    kept, _ = apply_filters(rows, meta, parse_filters("no_person"))
    with pytest.raises(ValueError):
        make_items(kept, bins=BINS, per_bin=3, images_per_item=3, splits=("explore", "confirm"))
