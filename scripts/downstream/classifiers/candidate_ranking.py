#!/usr/bin/env python3
"""
Classifier for 2-candidate Senate ranking experiment (apol→pol).

Parses narrative ranking responses to determine which of two 2026 Senate
candidates each persona ranked #1, then computes per-group alignment.

Usage: python -m scripts.downstream.classifiers.candidate_ranking
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.downstream.classifiers.common import (
    load_records, save_records, group_by_condition,
)

GROUP_ORDER_ATP = ["dem_impl", "rep_impl"]
GROUP_DISPLAY_ATP = {
    "dem_impl": "Dem(implicit)",
    "rep_impl": "Rep(implicit)",
}


def parse_candidate_ranking(
    response: str,
    dem_name: str,
    rep_name: str,
) -> Dict[str, Optional[int]]:
    """Parse which candidate was ranked #1 and #2.

    Strategy:
    1. Look for "#1", "#2", "1.", "2." patterns near candidate names
    2. Fall back to mention order
    """
    ranks: Dict[str, Optional[int]] = {dem_name: None, rep_name: None}
    text = response
    text_lower = text.lower()

    dem_lower = dem_name.lower()
    rep_lower = rep_name.lower()
    first_last_dem = dem_lower.split()
    first_last_rep = rep_lower.split()

    # Look for the first/last name near rankings
    candidates = [
        (dem_name, dem_lower, first_last_dem),
        (rep_name, rep_lower, first_last_rep),
    ]

    for full_name, name_lower, parts in candidates:
        # Try various "rank is 1" patterns near the name
        name_first = parts[0]
        name_last = parts[-1] if len(parts) > 1 else parts[0]

        # Patterns: "#1 ... Roy Cooper" or "Roy Cooper ... #1" or "1. Roy Cooper"
        patterns = [
            rf'#1\b.*?\b{re.escape(name_last)}',
            rf'rank\s*(?:ed)?\s*#1\b.*?\b{re.escape(name_last)}',
            rf'\b{re.escape(name_last)}\b.*?#1',
            rf'#1.*?\b{re.escape(name_first)}\b',
            rf'\b{re.escape(name_first)}\b.*?#1',
            rf'(?:^|\n)\s*1[\.\)]\s*.*?\b{re.escape(name_first)}',
        ]
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                ranks[full_name] = 1
                break

        if ranks[full_name] is not None:
            continue

        # Try rank #2 patterns
        patterns2 = [
            rf'#2\b.*?\b{re.escape(name_last)}',
            rf'\b{re.escape(name_last)}\b.*?#2',
            rf'#2.*?\b{re.escape(name_first)}\b',
            rf'\b{re.escape(name_first)}\b.*?#2',
            rf'(?:^|\n)\s*2[\.\)]\s*.*?\b{re.escape(name_first)}',
        ]
        for pat in patterns2:
            if re.search(pat, text, re.IGNORECASE):
                ranks[full_name] = 2
                break

    # Fill in the other rank if one is known
    if ranks[dem_name] == 1:
        ranks[rep_name] = 2
    elif ranks[rep_name] == 1:
        ranks[dem_name] = 2
    elif ranks[dem_name] == 2:
        ranks[rep_name] = 1
    elif ranks[rep_name] == 2:
        ranks[dem_name] = 1

    # Fallback: mention order
    if None in ranks.values():
        dem_pos = text_lower.find(dem_lower)
        rep_pos = text_lower.find(rep_lower)
        if dem_pos >= 0 and rep_pos >= 0:
            if dem_pos < rep_pos:
                ranks[dem_name] = 1
                ranks[rep_name] = 2
            else:
                ranks[rep_name] = 1
                ranks[dem_name] = 2

    return ranks


def classify_candidate_ranking(record: Dict[str, Any]) -> Dict[str, Any]:
    response = record.get("_response", "")
    race = record.get("_race", {})
    dem_name = race.get("dem_name", "")
    rep_name = race.get("rep_name", "")

    if not dem_name or not rep_name or response.startswith("[ERROR"):
        return {"dem_first": None, "rep_first": None, "parse_error": True}

    ranks = parse_candidate_ranking(response, dem_name, rep_name)

    dem_first = ranks.get(dem_name) == 1
    rep_first = ranks.get(rep_name) == 1

    return {
        "dem_first": dem_first,
        "rep_first": rep_first,
        "dem_rank": ranks.get(dem_name),
        "rep_rank": ranks.get(rep_name),
        "parse_error": (None in ranks.values()),
        "race_key": race.get("key", ""),
        "state": race.get("state", ""),
    }


def run_candidate_ranking_pipeline() -> None:
    data_dir = os.path.join(ROOT_DIR, "data", "lvis_persona")
    input_path = os.path.join(data_dir, "apol_to_pol_candidate_ranking.jsonl")
    output_path = os.path.join(data_dir, "apol_to_pol_candidate_ranking_classified.jsonl")

    if not os.path.exists(input_path):
        print(f"SKIP: input not found at {input_path}")
        return

    records = load_records(input_path)
    print(f"\n{'='*60}")
    print(f"  CANDIDATE RANKING CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Loaded {len(records)} records")

    for r in records:
        r["_candidate_classification"] = classify_candidate_ranking(r)

    save_records(records, output_path)
    print(f"Saved classified records to {output_path}")

    # Per-group analysis
    # Override GROUP_ORDER for this experiment (only 2 groups)
    grouped: Dict[str, List[Dict]] = {g: [] for g in GROUP_ORDER_ATP}
    for r in records:
        g = r.get("group", "")
        if g in grouped:
            grouped[g].append(r)

    # Race distribution
    race_counts: Counter = Counter()
    for r in records:
        race_counts[r.get("_candidate_classification", {}).get("state", "?")] += 1
    print(f"\n  --- Race distribution ---")
    for state, cnt in race_counts.most_common():
        print(f"    {state}: {cnt}")

    # Per-group Dem-first proportion
    print(f"\n  --- Candidate alignment ---")
    for g in GROUP_ORDER_ATP:
        n = len(grouped[g])
        dem_first = sum(1 for r in grouped[g]
                        if r.get("_candidate_classification", {}).get("dem_first") is True)
        rep_first = sum(1 for r in grouped[g]
                        if r.get("_candidate_classification", {}).get("rep_first") is True)
        err = sum(1 for r in grouped[g]
                  if r.get("_candidate_classification", {}).get("parse_error") is True)
        dem_pct = f"{dem_first / max(n - err, 1) * 100:.1f}%" if n > err else "n/a"
        print(f"    {GROUP_DISPLAY_ATP[g]:<20s} "
              f"Dem #1: {dem_first}/{n} ({dem_pct})  "
              f"Rep #1: {rep_first}/{n}  Err: {err}/{n}")

    # By race
    print(f"\n  --- Alignment by race ---")
    for race_key in sorted(set(
        r.get("_candidate_classification", {}).get("race_key", "")
        for r in records
    )):
        race_recs = [r for r in records
                     if r.get("_candidate_classification", {}).get("race_key") == race_key]
        for g in GROUP_ORDER_ATP:
            gr = [r for r in race_recs if r["group"] == g]
            n = len(gr)
            dem_first = sum(1 for r in gr
                            if r.get("_candidate_classification", {}).get("dem_first") is True)
            rep_first = sum(1 for r in gr
                            if r.get("_candidate_classification", {}).get("rep_first") is True)
            if n > 0:
                print(f"    {race_key:<18s} {GROUP_DISPLAY_ATP[g]:<20s} "
                      f"Dem: {dem_first}/{n}  Rep: {rep_first}/{n}")

    # Sample
    print(f"\n  --- Sample responses ---")
    for g in GROUP_ORDER_ATP:
        for r in grouped[g]:
            resp = r.get("_response", "")
            if resp and not resp.startswith("[ERROR"):
                cls = r.get("_candidate_classification", {})
                label = "Dem #1" if cls.get("dem_first") else "Rep #1"
                print(f"\n    [{GROUP_DISPLAY_ATP[g]}] {r['persona_id']} ({label})")
                print(f"    {resp[:400]}...")
                break
    print()


if __name__ == "__main__":
    run_candidate_ranking_pipeline()
