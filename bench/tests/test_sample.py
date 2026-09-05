"""Sampler: integrity-only default, strata, 3-images-same-stratum, explore/confirm disjoint."""

import random

import pytest

from bench.sample import (
    DEFAULT_FILTERS, SENSITIVITY_FILTERS, apply_filters, assign_strata,
    coco_id_from_record_name, make_items, parse_filters,
)

BINS = 10
PER_BIN = 30           # 30 images/stratum -> 10 items/stratum -> 5 explore + 5 confirm
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
            meta[coco_id] = {"n_objects": 3 + (i % 13), "categories": ["cup", "table"],
                             "category_counts": {"cup": 1, "table": 2 + (i % 13)}}
    return rows, meta


def test_coco_id_parsing():
    assert coco_id_from_record_name("train2017/000000000030.jpg") == 30


def test_default_filters_are_integrity_only():
    """Task F: no content filtering by default -- selecting on content conditions on
    variables that may be on the causal path, and every threshold is a free parameter."""
    assert DEFAULT_FILTERS == ""
    parsed = parse_filters(DEFAULT_FILTERS)
    assert not any(parsed.values()), parsed

    rows, meta = _pool(n_per_bin=1, bins=1)
    row = dict(rows[0])
    info = dict(meta[row["coco_id"]])
    info["categories"] = ["person", "signboard", "cup"]
    info["category_counts"] = {"person": 4, "signboard": 1, "cup": 1}
    info["n_objects"] = 40
    row["image_w"], row["image_h"] = 1600, 400        # extreme aspect
    row["num_image_tokens"] = 64                      # below the old floor of 300

    kept, dropped = apply_filters([row], {row["coco_id"]: info}, parse_filters(DEFAULT_FILTERS))
    assert len(kept) == 1 and not dropped, "nothing may be dropped for its content"
    assert kept[0]["n_persons"] == 4                  # it is annotated instead
    assert kept[0]["has_text_cat"] is True
    assert kept[0]["text_cats"] == ["signboard"]

    # and the same row is dropped once the sensitivity filters are asked for
    kept2, dropped2 = apply_filters([row], {row["coco_id"]: info},
                                    parse_filters(SENSITIVITY_FILTERS))
    assert kept2 == [] and sum(dropped2.values()) == 1


def test_integrity_filters_are_not_optional():
    rows, meta = _pool(n_per_bin=1, bins=1)
    broken = dict(rows[0]); broken["image_token_mismatch"] = 1
    empty = dict(rows[0]); empty["num_image_tokens"] = 0
    filters = parse_filters(DEFAULT_FILTERS)
    assert apply_filters([broken], meta, filters)[1]["integrity_token_mismatch"] == 1
    assert apply_filters([empty], meta, filters)[1]["integrity_no_image_tokens"] == 1


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
    assert one({}, {"categories": ["person", "cup"]})[1]["opt_person"] == 1
    assert one({}, {"categories": ["signboard"]})[1]["opt_text_category"] == 1
    assert one({}, {"categories": ["flag"]})[1]["opt_text_category"] == 1
    assert one({}, {"n_objects": 20})[1]["opt_n_objects"] == 1
    assert one({}, {"n_objects": 2})[1]["opt_n_objects"] == 1
    assert one({"image_w": 1600, "image_h": 480}, {})[1]["opt_aspect"] == 1
    assert one({"num_image_tokens": 100}, {})[1]["opt_token_count"] == 1
    assert one({"image_token_mismatch": 1}, {})[1]["integrity_token_mismatch"] == 1
    assert apply_filters([base], {}, parse_filters("no_person"))[1]["no_lvis_annotation"] == 1


def test_strata_are_equal_count_and_monotone():
    rows, _ = _pool()
    tagged = assign_strata(rows, bins=BINS)
    counts = {}
    for row in tagged:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    assert set(counts) == set(range(BINS))
    assert len(set(counts.values())) == 1, counts
    means = {}
    for row in tagged:
        means.setdefault(row["stratum"], []).append(row["image_mean"])
    ordered = [sum(v) / len(v) for _, v in sorted(means.items())]
    assert ordered == sorted(ordered)


def _items():
    rows, meta = _pool()
    kept, _ = apply_filters(rows, meta, parse_filters(DEFAULT_FILTERS))
    return make_items(kept, bins=BINS, per_bin=PER_BIN,
                      images_per_item=IMAGES_PER_ITEM, splits=("explore", "confirm"), seed=7)


def test_stratum_counts_are_exact():
    items, profile = _items()
    per_split = (PER_BIN // IMAGES_PER_ITEM) // 2
    for split in ("explore", "confirm"):
        assert len(items[split]) == per_split * BINS
        counts = {}
        for item in items[split]:
            counts[item["stratum"]] = counts.get(item["stratum"], 0) + 1
        assert counts == {d: per_split for d in range(BINS)}
    assert profile["n_items"] == {"explore": per_split * BINS, "confirm": per_split * BINS}


def test_three_images_per_item_all_from_the_same_stratum():
    items, _ = _items()
    rows, _meta = _pool()
    stratum_of = {r["record_name"]: r["stratum"] for r in assign_strata(rows, bins=BINS)}
    for split_items in items.values():
        for item in split_items:
            assert len(item["images"]) == IMAGES_PER_ITEM
            assert len(item["image_scores"]) == IMAGES_PER_ITEM
            assert {stratum_of[name] for name in item["images"]} == {item["stratum"]}


def test_items_declare_the_stratum_as_the_primary_iv_and_annotate_the_rest():
    """Task E: image_mean_mean is demoted to a covariate; the stratum is x."""
    items, profile = _items()
    assert profile["primary_iv"] == "stratum"
    for item in items["explore"]:
        assert item["primary_iv"] == "stratum"
        assert isinstance(item["stratum"], int)
        covariates = item["covariates"]
        assert "image_mean_mean" in covariates          # kept, but only as a covariate
        for annotation in ("n_persons", "has_text_cat", "n_objects", "aspect",
                           "num_image_tokens"):
            assert annotation in covariates, annotation
        assert len(covariates["n_persons"]) == IMAGES_PER_ITEM
        assert covariates["n_persons_total"] == sum(covariates["n_persons"])


def test_item_round_trips_through_the_item_type_and_reads_v1_files():
    from bench.types import Item
    items, _ = _items()
    item = Item.from_dict(items["explore"][0])
    assert item.primary_iv == "stratum"
    assert item.stratum == items["explore"][0]["stratum"]
    legacy = dict(items["explore"][0]); legacy["decile"] = legacy.pop("stratum")
    assert Item.from_dict(legacy).stratum == item.stratum


def test_profile_reports_the_annotation_shares_per_stratum():
    _items_out, profile = _items()
    assert len(profile["strata"]) == BINS
    for row in profile["strata"]:
        assert 0.0 <= row["share_with_person"] <= 1.0
        assert 0.0 <= row["share_with_text_cat"] <= 1.0
        assert row["top_categories"], "the confounding structure must be reported, not hidden"


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
    kept, _ = apply_filters(rows, meta, parse_filters(DEFAULT_FILTERS))
    with pytest.raises(ValueError):
        make_items(kept, bins=BINS, per_bin=3, images_per_item=3, splits=("explore", "confirm"))
