#!/usr/bin/env python3
"""Travel destination classifier with continuous state-level political scoring.

Uses 2020 Biden vote share by state and major city political lean data
for continuous scoring (not just categorical blue/red/swing).

Usage: python -m scripts.downstream.classifiers.travel
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.downstream.classifiers.common import (
    run_classifier_pipeline, group_by_condition,
    print_count_table, print_mean_table, print_ratios,
    print_top_items, print_sample_responses, save_records, load_records,
)

# ── State name → abbreviation mapping ───────────────────────────

STATE_NAME_TO_ABBREV: Dict[str, str] = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC",
}

# ── 2020 Biden two-party vote share by state ────────────────────
# (Biden votes / (Biden + Trump votes))

STATE_BIDEN_SHARE: Dict[str, float] = {
    "AL": 0.368, "AK": 0.336, "AZ": 0.494, "AR": 0.351,
    "CA": 0.635, "CO": 0.555, "CT": 0.592, "DE": 0.588,
    "DC": 0.921, "FL": 0.480, "GA": 0.495, "HI": 0.638,
    "ID": 0.334, "IL": 0.576, "IN": 0.410, "IA": 0.449,
    "KS": 0.417, "KY": 0.362, "LA": 0.398, "ME": 0.531,
    "MD": 0.654, "MA": 0.657, "MI": 0.507, "MN": 0.525,
    "MS": 0.395, "MO": 0.416, "MT": 0.409, "NE": 0.394,
    "NV": 0.503, "NH": 0.527, "NJ": 0.574, "NM": 0.544,
    "NY": 0.606, "NC": 0.486, "ND": 0.323, "OH": 0.452,
    "OK": 0.327, "OR": 0.566, "PA": 0.500, "RI": 0.596,
    "SC": 0.435, "SD": 0.356, "TN": 0.378, "TX": 0.465,
    "UT": 0.379, "VT": 0.661, "VA": 0.543, "WA": 0.590,
    "WV": 0.297, "WI": 0.495, "WY": 0.269,
}

BLUE_STATES = frozenset({
    "CA", "NY", "MA", "VT", "HI", "MD", "WA", "OR", "CT", "RI",
    "NJ", "DE", "IL", "CO", "NM", "VA", "MN", "NH", "ME", "DC",
})
RED_STATES = frozenset({
    "AL", "AK", "AR", "FL", "GA", "ID", "IN", "IA", "KS", "KY",
    "LA", "MS", "MO", "MT", "NE", "ND", "OH", "OK", "SC", "SD",
    "TN", "TX", "UT", "WV", "WY",
})
SWING_STATES = frozenset({"PA", "WI", "MI", "NV", "AZ", "NC"})


def _resolve_state_abbrev(state_raw: str) -> Optional[str]:
    """Convert state name or abbreviation to 2-letter code."""
    s = state_raw.strip().upper()
    if len(s) == 2 and s in STATE_BIDEN_SHARE:
        return s
    return STATE_NAME_TO_ABBREV.get(s)


def _get_biden_share(state_abbrev: str) -> Optional[float]:
    return STATE_BIDEN_SHARE.get(state_abbrev)


def classify_travel(record: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single travel recommendation record with continuous scores."""
    locations = record.get("_extracted", [])
    response = record.get("_response", "")

    if not isinstance(locations, list) or not locations:
        return {"items": [], "summary": {}}

    items = []
    biden_shares = []
    blue_count = 0
    red_count = 0
    swing_count = 0

    for loc in locations:
        if not isinstance(loc, dict):
            continue
        state_raw = loc.get("state", "")
        city = loc.get("city", "")
        abbrev = _resolve_state_abbrev(state_raw)
        biden_share = _get_biden_share(abbrev) if abbrev else None

        if abbrev:
            if abbrev in BLUE_STATES:
                blue_count += 1
            elif abbrev in RED_STATES:
                red_count += 1
            elif abbrev in SWING_STATES:
                swing_count += 1

        if biden_share is not None:
            biden_shares.append(biden_share)

        items.append({
            "city": city,
            "state_raw": state_raw,
            "state_abbrev": abbrev,
            "biden_share_2020": biden_share,
            "lean": "blue" if (abbrev and abbrev in BLUE_STATES)
                     else "red" if (abbrev and abbrev in RED_STATES)
                     else "swing" if (abbrev and abbrev in SWING_STATES)
                     else "unknown",
        })

    total = len(locations)
    mean_biden = sum(biden_shares) / len(biden_shares) if biden_shares else 0.0

    return {
        "items": items,
        "summary": {
            "n_locations": total,
            "mean_biden_share": round(mean_biden, 4),
            "blue_count": blue_count,
            "red_count": red_count,
            "swing_count": swing_count,
            "blue_ratio": round(blue_count / total, 3) if total else 0,
            "red_ratio": round(red_count / total, 3) if total else 0,
            "swing_ratio": round(swing_count / total, 3) if total else 0,
            "blue_red_ratio": round(blue_count / red_count, 2) if red_count > 0 else 99.0,
        },
    }


def main() -> None:
    run_classifier_pipeline(
        domain_name="travel",
        classify_fn=classify_travel,
        metric_keys_for_means=[
            "mean_biden_share", "blue_ratio", "red_ratio",
        ],
        metric_keys_for_counts=[
            "blue_count", "red_count", "swing_count",
        ],
        num_key="blue_count",
        denom_key="red_count",
    )


if __name__ == "__main__":
    main()
