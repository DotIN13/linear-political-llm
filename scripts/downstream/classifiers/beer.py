#!/usr/bin/env python3
"""Beer classifier for LVIS consumer preference recommendations.

Classifies bold beer recommendations by style, brewery type, and characteristics.

Usage: python -m scripts.downstream.classifiers.beer
"""

import os
import sys
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.downstream.classifiers.common import (
    run_classifier_pipeline, group_by_condition, load_records, save_records,
    print_count_table, print_mean_table, print_ratios,
    print_top_items, print_sample_responses,
)

# ── Beer style classification ──────────────────────────────────

# Style detection from beer name text
STYLE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("hazy_ipa", ["hazy", "neipa", "haze", "juicy"]),
    ("ipa", ["ipa", "india pale ale"]),
    ("double_ipa", ["double ipa", "dipa", "imperial ipa", "triple ipa"]),
    ("pale_ale", ["pale ale"]),
    ("lager", ["lager", "pilsner", "pils", "pilsener"]),
    ("light_lager", ["light lager", "light beer", "lite", "light"]),
    ("amber_ale", ["amber", "red ale", "red"]),
    ("wheat", ["wheat", "hefeweizen", "witbier", "wit", "weiss", "weizen"]),
    ("stout", ["stout"]),
    ("porter", ["porter"]),
    ("sour", ["sour", "gose", "berliner", "lambic", "wild ale", "farmhouse"]),
    ("saison", ["saison", "farmhouse ale"]),
    ("belgian", ["belgian", "dubbel", "tripel", "quad", "trappist", "abbey"]),
    ("brown_ale", ["brown ale", "nut brown"]),
    ("kolsch", ["kolsch", "kölsch"]),
    ("cider", ["cider"]),
    ("mexican_lager", ["mexican", "modelo", "corona", "pacifico", "dos equis"]),
]

# Brewery type classification
NATIONAL_MACRO = {
    "budweiser", "bud light", "bud", "miller", "coors", "busch",
    "pabst", "pbr", "natural light", "natty", "michelob", "keystone",
    "yuengling", "hamm's", "hamms", "old milwaukee", "genesee",
    "lone star", "rainier", "olympia", "narragansett",
}

NATIONAL_CRAFT = {
    "sierra nevada", "new belgium", "samuel adams", "sam adams",
    "boston beer", "lagunitas", "stone brewing", "stone",
    "dogfish head", "dogfish", "bell's brewery", "bells",
    "founders", "goose island", "brooklyn brewery", "brooklyn",
    "deschutes", "anchor brewing", "anchor", "ballast point",
    "firestone walker", "victory", "troegs", "odell",
    "cigar city", "oskars blues", "oskar blues",
    "sweetwater", "revision", "modern times",
}

REGIONAL_CRAFT = {
    "russian river", "alchemist", "tree house", "hill farmstead",
    "trillium", "other half", "three floyds", "surly", "summit",
    "left hand", "new holland", "shorts", "wicked weed",
    "burial", "creature comforts", "knee deep", "jolly pumpkin",
    "great divide", "green flash", "revolution brewing", "revolution",
    "epic brewing", "flying dog", "21st amendment",
    "half acre", "pipeworks", "toppling goliath",
}

IMPORT_BEER = {
    "heineken", "corona", "modelo", "stella artois", "guinness",
    "dos equis", "pacifico", "peroni", "moretti", "tsingtao",
    "asahi", "sapporo", "kirin", "becks", "st pauli",
    "fosters", "carlsberg", "amstel", "grolsch", "newcastle",
    "boddingtons", "bass", "hoegaarden", "leffe", "chimay",
    "duvel", "orval", "westmalle", "rochefort", "delirium",
    "paulaner", "weihenstephaner", "hofbrau", "spaten", "franziskaner",
}


def _detect_styles(item_lower: str) -> List[str]:
    """Detect beer styles from a bold item text."""
    styles = []
    for style_name, keywords in STYLE_PATTERNS:
        if any(kw in item_lower for kw in keywords):
            styles.append(style_name)
    if not styles:
        styles.append("other")
    return styles


def _detect_brewery_type(item_lower: str) -> str:
    """Detect brewery type from bold item text."""
    if any(b in item_lower for b in NATIONAL_MACRO):
        return "national_macro"
    if any(b in item_lower for b in IMPORT_BEER):
        return "import"
    if any(b in item_lower for b in REGIONAL_CRAFT):
        return "regional_craft"
    if any(b in item_lower for b in NATIONAL_CRAFT):
        return "national_craft"
    return "likely_craft"  # default assumption


def classify_beer(record: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single beer recommendation record."""
    bold_items = record.get("_extracted", [])
    response = record.get("_response", "")

    if not bold_items:
        return {"items": [], "summary": {}}

    items = []
    style_counts: Dict[str, int] = {}
    macro_count = 0
    import_count = 0
    national_craft_count = 0
    regional_craft_count = 0

    for item in bold_items:
        item_lower = item.lower()
        styles = _detect_styles(item_lower)
        brewery = _detect_brewery_type(item_lower)

        for s in styles:
            style_counts[s] = style_counts.get(s, 0) + 1

        if brewery == "national_macro":
            macro_count += 1
        elif brewery == "import":
            import_count += 1
        elif brewery == "national_craft":
            national_craft_count += 1
        elif brewery == "regional_craft":
            regional_craft_count += 1

        items.append({
            "name": item,
            "styles": styles,
            "brewery_type": brewery,
        })

    total = len(bold_items)
    ipa_count = style_counts.get("ipa", 0) + style_counts.get("hazy_ipa", 0) + style_counts.get("double_ipa", 0)
    lager_count = style_counts.get("lager", 0) + style_counts.get("light_lager", 0) + style_counts.get("mexican_lager", 0)
    stout_count = style_counts.get("stout", 0) + style_counts.get("porter", 0)
    pale_ale_count = style_counts.get("pale_ale", 0)
    amber_count = style_counts.get("amber_ale", 0)
    wheat_count = style_counts.get("wheat", 0)

    return {
        "items": items,
        "summary": {
            "n_items": total,
            "ipa_ratio": round(ipa_count / total, 3) if total else 0,
            "lager_ratio": round(lager_count / total, 3) if total else 0,
            "stout_porter_ratio": round(stout_count / total, 3) if total else 0,
            "pale_ale_ratio": round(pale_ale_count / total, 3) if total else 0,
            "wheat_ratio": round(wheat_count / total, 3) if total else 0,
            "ipa_count": ipa_count,
            "lager_count": lager_count,
            "macro_count": macro_count,
            "import_count": import_count,
            "national_craft_count": national_craft_count,
            "regional_craft_count": regional_craft_count,
            "macro_ratio": round(macro_count / total, 3) if total else 0,
        },
        "style_counts": style_counts,
    }


def main() -> None:
    run_classifier_pipeline(
        domain_name="beer",
        classify_fn=classify_beer,
        metric_keys_for_means=[
            "ipa_ratio", "lager_ratio", "stout_porter_ratio",
            "pale_ale_ratio", "macro_ratio",
        ],
        metric_keys_for_counts=[
            "ipa_count", "lager_count", "macro_count",
            "national_craft_count", "regional_craft_count",
        ],
        num_key="ipa_count",
        denom_key="lager_count",
    )


if __name__ == "__main__":
    main()
