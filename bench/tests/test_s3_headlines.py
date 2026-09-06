"""Self-check for bench/data/s3_headlines_v1.json.

The file is a hand-curated 12-headline set (6 topics x 2 sides, each topic carrying
one left-slanted and one right-slanted outlet). These tests assert its integrity:
the item counts, uniqueness, the topic x side crossing, and above all that every
``slant`` matches ``data/adfontesmedia.csv`` exactly (it was transcribed by hand).
"""

import csv
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

HERE = Path(__file__).resolve()
DATA_FILE = HERE.parent.parent / "data" / "s3_headlines_v1.json"
CSV_FILE = HERE.parent.parent.parent / "data" / "adfontesmedia.csv"

LOOSE_HOST_MAP = {
    "Slate": "slate.com",
    "Fox News (website)": "foxnews.com",
    "NPR (website)": "npr.org",
    "Washington Examiner": "washingtonexaminer.com",
    "HuffPost": "huffpost.com",
    "Daily Caller": "dailycaller.com",
    "NBC News (website)": "nbcnews.com",
    "Daily Wire": "dailywire.com",
    "CNN (website)": "cnn.com",
    "The Epoch Times": "theepochtimes.com",
    "MSNBC (website)": "msnbc.com",
    "Fox Business (website)": "foxbusiness.com",
}


@pytest.fixture(scope="module")
def dataset():
    with DATA_FILE.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def bias_lookup():
    with CSV_FILE.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["", "moniker_name", "bias_mean"]
    return {r[1]: r[2] for r in rows[1:]}


def parse_date(value):
    """Accept either ``YYYY-MM`` (-> first of month) or ``YYYY-MM-DD``."""
    parts = value.split("-")
    if len(parts) == 2:
        y, m = parts
        return (int(y), int(m), 1)
    if len(parts) == 3:
        return tuple(int(p) for p in parts)
    raise ValueError(f"unparseable date {value!r}")


def test_n_items_and_headline_count(dataset):
    assert dataset["n_items"] == 12
    assert len(dataset["headlines"]) == 12


def test_hids_and_outlets_are_all_unique(dataset):
    headlines = dataset["headlines"]
    hids = [h["hid"] for h in headlines]
    outlets = [h["outlet"] for h in headlines]
    assert len(hids) == len(set(hids)) == 12
    assert len(outlets) == len(set(outlets)) == 12


def test_topics_are_crossed_with_side(dataset):
    headlines = dataset["headlines"]
    topics = [h["topic"] for h in headlines]
    assert len(set(topics)) == 6
    for topic in set(topics):
        rows = [h for h in headlines if h["topic"] == topic]
        assert len(rows) == 2, f"topic {topic!r} has {len(rows)} items, expected 2"
        sides = sorted(h["side"] for h in rows)
        assert sides == ["left", "right"], f"topic {topic!r} sides are {sides}"


def test_every_slant_matches_adfontesmedia_exactly(dataset, bias_lookup):
    mismatches = []
    for h in dataset["headlines"]:
        outlet = h["outlet"]
        if outlet not in bias_lookup:
            mismatches.append((outlet, "missing from CSV"))
            continue
        csv_value = bias_lookup[outlet]
        if float(csv_value) != h["slant"]:
            mismatches.append((outlet, csv_value, h["slant"]))
    assert not mismatches, f"slant/CSV mismatches: {mismatches}"


def test_left_slant_negative_right_slant_positive(dataset):
    for h in dataset["headlines"]:
        if h["side"] == "left":
            assert h["slant"] < 0, f"{h['hid']} left slant {h['slant']} not < 0"
        else:
            assert h["slant"] > 0, f"{h['hid']} right slant {h['slant']} not > 0"


def test_centered_slant_consistency(dataset):
    mean = dataset["set_mean_slant"]
    centered = []
    for h in dataset["headlines"]:
        assert h["slant_c"] == pytest.approx(h["slant"] - mean, abs=1e-12)
        centered.append(h["slant_c"])
    assert sum(centered) == pytest.approx(0.0, abs=1e-9)


def test_set_mean_slant_is_the_mean(dataset):
    slants = [h["slant"] for h in dataset["headlines"]]
    assert dataset["set_mean_slant"] == pytest.approx(sum(slants) / len(slants), abs=1e-12)


def test_fields_are_nonempty_and_dates_in_range(dataset):
    for h in dataset["headlines"]:
        assert h["headline"].strip(), f"{h['hid']} has empty headline"
        assert h["url"].strip(), f"{h['hid']} has empty url"
        assert h["date"].strip(), f"{h['hid']} has empty date"
        d = parse_date(h["date"])
        assert (2026, 8, 1) <= d <= (2026, 9, 6), f"{h['hid']} date {h['date']!r} out of range"


def test_url_host_matches_outlet(dataset):
    for h in dataset["headlines"]:
        host = urlparse(h["url"]).netloc
        expected = LOOSE_HOST_MAP.get(h["outlet"])
        assert expected, f"no host mapping for outlet {h['outlet']!r}"
        assert host.endswith(expected), f"{h['outlet']!r} url host {host!r} !~ {expected!r}"
