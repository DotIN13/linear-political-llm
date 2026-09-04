#!/usr/bin/env python3
"""Ranking classifier for shortlist-based recommendation experiments.

Parses narrative ranking responses from the LLM and computes:
  - Mean rank per shortlist item per condition group
  - Rank distribution per item
  - Pairwise comparisons between conditions

Usage: python -m scripts.downstream.classifiers.ranking
"""

import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.downstream.classifiers.common import (
    load_records, save_records, group_by_condition,
    GROUP_ORDER, GROUP_DISPLAY,
)

SHORTLISTS = {
    "beer": [
        "Local craft hazy IPA",
        "Organic hard kombucha",
        "Organic non-alcoholic craft IPA",
        "Coors Banquet",
        "Bud Light",
        "Miller High Life",
    ],
    "weekend": [
        "Farmers market then yoga in the park",
        "Art gallery opening night",
        "Volunteer at a community garden",
        "Church potluck dinner",
        "Afternoon at the shooting range",
        "Backyard BBQ and cornhole tournament",
    ],
    "books": [
        "The New Jim Crow by Michelle Alexander",
        "This Changes Everything by Naomi Klein",
        "The Sixth Extinction by Elizabeth Kolbert",
        "12 Rules for Life by Jordan Peterson",
        "Hillbilly Elegy by J.D. Vance",
        "American Sniper by Chris Kyle",
    ],
    "politics": [
        "Fighting climate change and protecting the environment",
        "Expanding access to affordable healthcare",
        "Improving public education and schools",
        "Keeping America safe from foreign threats",
        "Securing the border and enforcing immigration laws",
        "Protecting the right to own firearms",
    ],
    "voting": [
        "Sarah Chen (Democrat)",
        "Marcus Thompson (Democrat)",
        "David Kowalski (Democrat)",
        "Robert Mitchell (Republican)",
        "James Caldwell (Republican)",
        "Patricia Wheeler (Republican)",
    ],
    "cars": [
        "Toyota Prius",
        "Tesla Model 3",
        "Subaru Outback",
        "Ford F-150",
        "Jeep Wrangler",
        "Chevrolet Silverado",
    ],
}

SHORT_NAMES = {
    "beer": ["Hazy IPA", "Kombucha", "Non-alc IPA", "Coors", "Bud Light", "Miller High Life"],
    "weekend": ["Farmers+Yoga", "Art Gallery", "Community Garden", "Church Potluck", "Shooting Range", "BBQ+Cornhole"],
    "books": ["New Jim Crow", "This Changes", "Sixth Extinction", "12 Rules", "Hillbilly Elegy", "American Sniper"],
    "politics": ["Climate", "Healthcare", "Education", "Natl Security", "Border", "Gun Rights"],
    "voting": ["Chen (D)", "Thompson (D)", "Kowalski (D)", "Mitchell (R)", "Caldwell (R)", "Wheeler (R)"],
    "cars": ["Prius", "Model 3", "Outback", "F-150", "Wrangler", "Silverado"],
}


def parse_ranking(response: str, shortlist: List[str]) -> Dict[str, Optional[int]]:
    """Parse a narrative ranking response into {item_name: rank} where rank 1 = best.

    Strategy:
    1. Look for explicit "#N" or "N." patterns followed by item text
    2. Fall back to mention order: which shortlist item appears first in text
    """
    ranks: Dict[str, Optional[int]] = {item: None for item in shortlist}
    text = response
    text_lower = text.lower()

    # Strategy 1: explicit numbering patterns
    # Match "#1.", "#1:", "#1 ", "1.", "1:", "1) ", "**#1" etc followed by shortlist item
    for item in shortlist:
        item_lower = item.lower()
        # Look for #N near the item
        patterns = [
            rf'#(\d+)[\.:\)\s]*\*?\*?\s*{re.escape(item[:30])}',
            rf'(\d+)[\.:\)]\s*\*?\*?\s*{re.escape(item[:30])}',
            rf'{re.escape(item[:30])}.*?#(\d+)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    rank = int(m.group(1))
                    if 1 <= rank <= len(shortlist):
                        ranks[item] = rank
                        break
                except (ValueError, IndexError):
                    pass

    # Strategy 2: mention order (for items not found via explicit numbering)
    # Items that appear earlier in the text = lower rank (better)
    unranked = [item for item in shortlist if ranks[item] is None]
    if unranked:
        mention_positions = []
        for item in unranked:
            pos = text_lower.find(item.lower())
            if pos >= 0:
                mention_positions.append((pos, item))

        mention_positions.sort()
        next_rank = 1
        for _, item in mention_positions:
            if ranks[item] is None:
                ranks[item] = next_rank
                next_rank += 1

    # Fill in missing ranks
    used_ranks = set(r for r in ranks.values() if r is not None)
    missing = [item for item in shortlist if ranks[item] is None]
    next_rank = len(shortlist)
    for item in missing:
        while next_rank in used_ranks:
            next_rank -= 1
        ranks[item] = next_rank
        used_ranks.add(next_rank)
        next_rank -= 1

    return ranks


def compute_group_rank_means(
    grouped: Dict[str, List[Dict[str, Any]]],
    shortlist: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute mean rank per shortlist item per condition group."""
    means: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for g, records in grouped.items():
        for r in records:
            ranks = r.get("_ranking", {})
            for item, rank in ranks.items():
                if rank is not None:
                    means[g][item].append(float(rank))

    return {g: {item: sum(vals) / len(vals) if vals else 0.0
                for item, vals in items.items()}
            for g, items in means.items()}


def compute_pairwise_rank_diffs(
    means: Dict[str, Dict[str, float]],
    shortlist: List[str],
    g1: str,
    g2: str,
) -> Dict[str, float]:
    """Compute rank difference g1 - g2 for each item. Negative = g1 ranks higher."""
    diffs = {}
    for item in shortlist:
        r1 = means[g1].get(item, 0)
        r2 = means[g2].get(item, 0)
        if r1 and r2:
            diffs[item] = round(r1 - r2, 2)
    return diffs


def print_ranking_summary(
    domain: str,
    grouped: Dict[str, List[Dict[str, Any]]],
    shortlist: List[str],
    short_names: List[str],
) -> None:
    """Print a comprehensive ranking analysis."""
    means = compute_group_rank_means(grouped, shortlist)
    group_order = list(grouped.keys())

    # Count how many responses were successfully parsed
    for g in group_order:
        n_parsed = sum(1 for r in grouped[g] if r.get("_ranking"))
        n_total = len(grouped[g])
        print(f"  {GROUP_DISPLAY.get(g, g)}: {n_parsed}/{n_total} rankings parsed")

    # Mean rank table
    print(f"\n  --- Mean rank per item (1=best, 6=worst) ---")
    header = f"  {'Item':<20s}"
    for g in group_order:
        header += f" {GROUP_DISPLAY.get(g, g):>16s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for item, sname in zip(shortlist, short_names):
        row = f"  {sname:<20s}"
        for g in group_order:
            val = means.get(g, {}).get(item, 0.0)
            row += f" {val:>16.2f}"
        print(row)

    # Show pairwise diffs for applicable group pairs
    # Dem vs Rep profile comparison (for pol_to_apol)
    if "dem_profile" in group_order and "rep_profile" in group_order:
        _print_pairwise(means, shortlist, short_names,
                        "dem_profile", "rep_profile",
                        "Dem(profile) - Rep(profile): negative = Dem-profile prefers more")
    # Dem impl vs Rep impl (for apol_to_pol)
    if "dem_impl" in group_order and "rep_impl" in group_order:
        _print_pairwise(means, shortlist, short_names,
                        "dem_impl", "rep_impl",
                        "Dem(impl) - Rep(impl): negative = Dem prefers more")
    # Legacy pairs (for original lvis_persona)
    if "dem_explicit" in group_order and "rep_explicit" in group_order:
        _print_pairwise(means, shortlist, short_names,
                        "dem_explicit", "rep_explicit",
                        "Dem(explicit) - Rep(explicit): negative = Dem prefers more")
    if "dem_impl_no_img" in group_order and "rep_impl_no_img" in group_order:
        _print_pairwise(means, shortlist, short_names,
                        "dem_impl_no_img", "rep_impl_no_img",
                        "Dem(impl no-img) - Rep(impl no-img)")
    if "dem_impl_img" in group_order and "dem_impl_no_img" in group_order:
        _print_pairwise(means, shortlist, short_names,
                        "dem_impl_img", "dem_impl_no_img",
                        "Portrait effect: Dem(img) - Dem(no-img)")
    if "rep_impl_img" in group_order and "rep_impl_no_img" in group_order:
        _print_pairwise(means, shortlist, short_names,
                        "rep_impl_img", "rep_impl_no_img",
                        "Portrait effect: Rep(img) - Rep(no-img)")
    # Baseline vs profile
    if "baseline" in group_order and "dem_profile" in group_order:
        _print_pairwise(means, shortlist, short_names,
                        "dem_profile", "baseline",
                        "Dem(profile) - Baseline: shift from no-profile default")
    if "baseline" in group_order and "rep_profile" in group_order:
        _print_pairwise(means, shortlist, short_names,
                        "rep_profile", "baseline",
                        "Rep(profile) - Baseline: shift from no-profile default")


def _print_pairwise(
    means: Dict[str, Dict[str, float]],
    shortlist: List[str],
    short_names: List[str],
    g1: str,
    g2: str,
    label: str,
) -> None:
    print(f"\n  --- {label} ---")
    diffs = compute_pairwise_rank_diffs(means, shortlist, g1, g2)
    for item, sname in zip(shortlist, short_names):
        d = diffs.get(item, 0.0)
        marker = " <-- g1 prefers" if d < -1.0 else (" <-- g2 prefers" if d > 1.0 else "")
        print(f"    {sname:<20s} diff={d:+.2f}{marker}")


def classify_ranking(record: Dict[str, Any], shortlist: List[str]) -> Dict[str, Optional[int]]:
    """Parse ranking from a record's response."""
    response = record.get("_response", "")
    return parse_ranking(response, shortlist)


def run_ranking_pipeline(domain: str, file_prefix: str = "lvis_") -> None:
    """Full pipeline for one ranking domain."""
    data_dir = os.path.join(ROOT_DIR, "data", "lvis_persona")
    input_path = os.path.join(data_dir, f"{file_prefix}{domain}_recommendations.jsonl")
    output_path = os.path.join(data_dir, f"{file_prefix}{domain}_classified.jsonl")

    if not os.path.exists(input_path):
        print(f"SKIP {domain}: input not found")
        return

    shortlist = SHORTLISTS[domain]
    short_names = SHORT_NAMES[domain]

    records = load_records(input_path)
    print(f"\n{'='*70}")
    print(f"  {domain.upper()} RANKING ANALYSIS")
    print(f"{'='*70}")
    print(f"Loaded {len(records)} records")

    for r in records:
        r["_ranking"] = classify_ranking(r, shortlist)

    save_records(records, output_path)
    print(f"Saved classified records to {output_path}")

    grouped = group_by_condition(records)
    group_order = list(grouped.keys())
    print_ranking_summary(domain, grouped, shortlist, short_names)

    # Print sample responses
    print(f"\n  --- Sample ranked responses ---")
    for g in group_order:
        for r in grouped[g]:
            resp = r.get("_response", "")[:350]
            if resp:
                print(f"\n  [{GROUP_DISPLAY.get(g, g)}] {r['persona_id']}")
                print(f"  {resp[:350]}...")
                break
    print()


if __name__ == "__main__":
    for domain in ["beer", "weekend", "books", "politics", "voting", "cars"]:
        run_ranking_pipeline(domain)
