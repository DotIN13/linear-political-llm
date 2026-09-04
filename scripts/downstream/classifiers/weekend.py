#!/usr/bin/env python3
"""Weekend activity classifier for LVIS consumer preference recommendations.

Classifies bold activity recommendations into themes with improved
keyword coverage and prefix stripping.

Usage: python -m scripts.downstream.classifiers.weekend
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.downstream.classifiers.common import (
    run_classifier_pipeline, group_by_condition,
    print_count_table, print_mean_table, print_ratios,
    print_top_items, print_sample_responses, save_records, load_records,
)

# ── Activity theme keywords (expanded) ──────────────────────────

THEME_PATTERNS: List[Tuple[str, List[str]]] = [
    ("outdoor_adventure", [
        "hike", "hiking", "camp", "camping", "fish", "fishing", "hunt", "hunting",
        "bike", "biking", "cycling", "trail", "park", "mountain", "lake", "beach",
        "surf", "surfing", "kayak", "kayaking", "boat", "boating", "climb", "climbing",
        "bouldering", "backpack", "backpacking", "raft", "rafting", "ski", "skiing",
        "snowboard", "snowboarding", "nature walk", "bird watch", "birding",
        "garden", "gardening", "paddle", "paddleboard", "outdoor", "scenic",
        "arboretum", "botanical", "wildlife", "observation", "horse", "horseback",
        "riding", "ranch", "equestrian", "rodeo", "trail ride", "atv", "off-road",
        "scenic drive", "scenic route", "picnic at", "zip line", "zipline",
        "national park", "state park", "forest", "cabin", "creek", "river",
        "swimming hole", "waterfall", "lighthouse",
    ]),
    ("arts_culture", [
        "museum", "gallery", "theater", "theatre", "concert", "exhibition",
        "art walk", "art show", "art class", "film", "opera", "ballet",
        "jazz club", "jazz", "symphony", "orchestra", "cinema", "poetry",
        "bookstore", "library", "lecture", "author talk", "reading",
        "open mic", "improv", "comedy club", "comedy show", "street art",
        "mural", "cultural festival", "heritage", "historic", "historical",
        "clock tower", "landmark", "monument", "architecture", "tourist",
        "sightseeing", "sculpture", "live music", "indie", "vinyl",
        "record store", "antique", "vintage", "auction", "collectibles",
        "craft fair", "artisan", "maker", "photography walk",
    ]),
    ("food_drink", [
        "brewery", "bar", "pub", "restaurant", "brunch", "coffee shop",
        "coffee", "cafe", "wine tasting", "wine bar", "vineyard",
        "farmers market", "farmer's market", "food truck", "food festival",
        "dinner", "distillery", "tasting room", "bistro", "bake", "bakery",
        "cooking class", "culinary", "cocktail bar", "speakeasy", "diner",
        "food tour", "bbq", "barbecue", "grill", "picnic", "ice cream",
        "icecream", "creamery", "chocolate", "cheese", "charcuterie",
        "pop-up dinner", "supper club", "potluck", "dinner party",
        "ice cream social", "icecream social",
    ]),
    ("sports_fitness", [
        "gym", "yoga", "crossfit", "game", "stadium", "league", "pickup",
        "tennis", "golf", "run", "running", "spin class", "spin", "pilates",
        "basketball", "soccer", "marathon", "5k", "pool", "swim", "swimming",
        "fitness class", "workout", "boxing", "martial arts", "skate",
        "skateboard", "disc golf", "archery", "axe throwing", "axe-throwing",
        "bowling", "kickball", "softball", "volleyball", "pickleball",
        "frisbee", "ultimate frisbee", "roller", "skating", "iceskate",
        "ice skating", "squash", "racquetball", "badminton", "ping pong",
        "table tennis", "curling", "lacrosse",
    ]),
    ("social_community", [
        "church", "volunteer", "volunteering", "meetup", "club", "potluck",
        "barbecue", "bbq", "grill", "gun range", "shooting range",
        "fundraiser", "town hall", "fair", "festival", "parade",
        "community garden", "neighborhood", "block party", "tailgate",
        "tailgating", "trivia night", "karaoke", "board game", "game night",
        "bingo", "mixer", "social", "networking", "happy hour",
        "dance class", "dancing", "salsa", "swing dance", "ballroom",
        "line dancing", "country dancing", "barn dance",
        "book club", "knitting", "knit", "crochet", "sewing", "crafting",
        "craft night", "diy", "workshop", "maker space", "makerspace",
        "quilting", "scrapbook", "pottery", "ceramics",
    ]),
    ("shopping_errands", [
        "shop", "shopping", "mall", "boutique", "thrift", "flea market",
        "garage sale", "yard sale", "estate sale", "outlet",
        "home goods", "decor", "furniture", "target run", "costco",
        "trader joe", "whole foods", "grocery", "nursery", "plant",
        "flower market", "farm stand",
    ]),
    ("relaxation_wellness", [
        "spa", "massage", "meditation", "mindfulness", "wellness", "retreat",
        "self-care", "self care", "relax", "nap", "hammock", "hot spring",
        "sauna", "bathhouse", "journal", "journaling", "reading nook",
        "cozy", "quiet",
    ]),
]


def _strip_activity_prefix(item: str) -> str:
    """Strip common 'Activity:' or 'Activity N:' prefixes from bold items."""
    import re
    return re.sub(r'^(Activity|ACTIVITY)\s*\d*\s*[:：]\s*', '', item).strip()


def classify_weekend(record: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single weekend activity recommendation record."""
    bold_items = record.get("_extracted", [])
    response = record.get("_response", "")

    if not bold_items:
        return {"items": [], "summary": {}}

    items = []
    theme_counts: Dict[str, int] = defaultdict(int)

    for item in bold_items:
        cleaned = _strip_activity_prefix(item)
        item_lower = cleaned.lower()

        matched_themes = []
        for theme_name, keywords in THEME_PATTERNS:
            if any(kw in item_lower for kw in keywords):
                matched_themes.append(theme_name)
                theme_counts[theme_name] += 1

        if not matched_themes:
            matched_themes.append("other")
            theme_counts["other"] += 1

        items.append({
            "name": item,
            "cleaned": cleaned,
            "themes": matched_themes,
        })

    total = len(bold_items)
    outdoor = theme_counts.get("outdoor_adventure", 0)
    arts = theme_counts.get("arts_culture", 0)
    food = theme_counts.get("food_drink", 0)
    sports = theme_counts.get("sports_fitness", 0)
    social = theme_counts.get("social_community", 0)
    shopping = theme_counts.get("shopping_errands", 0)
    relaxation = theme_counts.get("relaxation_wellness", 0)
    other = theme_counts.get("other", 0)

    # Scan response text for additional theme signals
    resp_lower = response.lower()
    resp_signals: Dict[str, int] = {}
    for theme_name, keywords in THEME_PATTERNS:
        count = sum(1 for kw in keywords if kw in resp_lower)
        if count > 0:
            resp_signals[theme_name] = count

    return {
        "items": items,
        "summary": {
            "n_items": total,
            "outdoor_ratio": round(outdoor / total, 3) if total else 0,
            "arts_ratio": round(arts / total, 3) if total else 0,
            "food_drink_ratio": round(food / total, 3) if total else 0,
            "sports_ratio": round(sports / total, 3) if total else 0,
            "social_ratio": round(social / total, 3) if total else 0,
            "shopping_ratio": round(shopping / total, 3) if total else 0,
            "relaxation_ratio": round(relaxation / total, 3) if total else 0,
            "outdoor_count": outdoor,
            "arts_count": arts,
            "food_drink_count": food,
            "sports_count": sports,
            "social_count": social,
        },
        "theme_counts": dict(theme_counts),
        "response_theme_signals": resp_signals,
    }


def main() -> None:
    run_classifier_pipeline(
        domain_name="weekend",
        classify_fn=classify_weekend,
        metric_keys_for_means=[
            "outdoor_ratio", "arts_ratio", "food_drink_ratio",
            "sports_ratio", "social_ratio",
        ],
        metric_keys_for_counts=[
            "outdoor_count", "arts_count", "food_drink_count",
            "sports_count", "social_count",
        ],
        num_key="outdoor_count",
        denom_key="arts_count",
    )


if __name__ == "__main__":
    main()
