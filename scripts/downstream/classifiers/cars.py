#!/usr/bin/env python3
"""Car/vehicle classifier for LVIS consumer preference recommendations.

Classifies each bold car recommendation by:
  - segment (subcompact..heavy_truck)
  - powertrain (ev..diesel)
  - green_score (0-100, higher = greener)
  - truck_score (0-100, higher = more truck-like)
  - nationality (domestic/import)
  - luxury flag
  - justification keywords

Usage: python -m scripts.downstream.classifiers.cars
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.downstream.classifiers.common import (
    run_classifier_pipeline, load_records, group_by_condition,
    print_count_table, print_mean_table, print_ratios,
    print_top_items, print_sample_responses, save_records,
)

# ── Car model database ──────────────────────────────────────────
# (make, model_segment) → {segment, powertrain, green_score, truck_score, nationality, luxury}

CAR_DB: Dict[str, Dict[str, Any]] = {}
DB_PREFIXES: Dict[str, Dict[str, Any]] = {}  # brand-level defaults for unknown models

def _register(make_defaults, *entries):
    DB_PREFIXES[make_defaults[0]] = make_defaults[1]
    for model_key, attrs in entries:
        CAR_DB[model_key] = attrs

# Brand defaults and known models
# format: _register(("brand", {default_attrs}), ("model", {attrs}), ...)

_register(
    ("toyota", {"nationality": "import", "green_score": 35, "truck_score": 20, "luxury": False}),
    ("toyota prius", {"segment": "compact", "powertrain": "hybrid", "green_score": 90, "truck_score": 0}),
    ("toyota prius prime", {"segment": "compact", "powertrain": "phev", "green_score": 95, "truck_score": 0}),
    ("toyota rav4", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 25}),
    ("toyota rav4 hybrid", {"segment": "crossover", "powertrain": "hybrid", "green_score": 65, "truck_score": 25}),
    ("toyota rav4 prime", {"segment": "crossover", "powertrain": "phev", "green_score": 75, "truck_score": 25}),
    ("toyota camry", {"segment": "midsize", "powertrain": "gas", "green_score": 40, "truck_score": 0}),
    ("toyota camry hybrid", {"segment": "midsize", "powertrain": "hybrid", "green_score": 70, "truck_score": 0}),
    ("toyota corolla", {"segment": "compact", "powertrain": "gas", "green_score": 45, "truck_score": 0}),
    ("toyota corolla hybrid", {"segment": "compact", "powertrain": "hybrid", "green_score": 75, "truck_score": 0}),
    ("toyota tacoma", {"segment": "truck", "powertrain": "gas", "green_score": 5, "truck_score": 80}),
    ("toyota tundra", {"segment": "truck", "powertrain": "gas", "green_score": 3, "truck_score": 90}),
    ("toyota 4runner", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 85}),
    ("toyota highlander", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 50}),
    ("toyota sequoia", {"segment": "suv", "powertrain": "gas", "green_score": 5, "truck_score": 80}),
    ("toyota sienna", {"segment": "van", "powertrain": "hybrid", "green_score": 50, "truck_score": 10}),
    ("toyota bz4x", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 30}),
)

_register(
    ("honda", {"nationality": "import", "green_score": 40, "truck_score": 15, "luxury": False}),
    ("honda civic", {"segment": "compact", "powertrain": "gas", "green_score": 50, "truck_score": 0}),
    ("honda accord", {"segment": "midsize", "powertrain": "gas", "green_score": 45, "truck_score": 0}),
    ("honda accord hybrid", {"segment": "midsize", "powertrain": "hybrid", "green_score": 70, "truck_score": 0}),
    ("honda cr-v", {"segment": "crossover", "powertrain": "gas", "green_score": 40, "truck_score": 20}),
    ("honda cr-v hybrid", {"segment": "crossover", "powertrain": "hybrid", "green_score": 65, "truck_score": 20}),
    ("honda hr-v", {"segment": "crossover", "powertrain": "gas", "green_score": 40, "truck_score": 15}),
    ("honda pilot", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 50}),
    ("honda ridgeline", {"segment": "truck", "powertrain": "gas", "green_score": 15, "truck_score": 70}),
    ("honda odyssey", {"segment": "van", "powertrain": "gas", "green_score": 25, "truck_score": 10}),
    ("honda prologue", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
)

_register(
    ("subaru", {"nationality": "import", "green_score": 35, "truck_score": 25, "luxury": False}),
    ("subaru outback", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 30}),
    ("subaru crosstrek", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 20}),
    ("subaru forester", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 25}),
    ("subaru ascent", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 35}),
    ("subaru impreza", {"segment": "compact", "powertrain": "gas", "green_score": 40, "truck_score": 5}),
    ("subaru legacy", {"segment": "midsize", "powertrain": "gas", "green_score": 35, "truck_score": 5}),
    ("subaru solterra", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
)

_register(
    ("mazda", {"nationality": "import", "green_score": 40, "truck_score": 10, "luxury": False}),
    ("mazda cx-30", {"segment": "crossover", "powertrain": "gas", "green_score": 40, "truck_score": 15}),
    ("mazda cx-5", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 20}),
    ("mazda cx-50", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 25}),
    ("mazda cx-9", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 40}),
    ("mazda cx-90", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 40}),
    ("mazda 3", {"segment": "compact", "powertrain": "gas", "green_score": 45, "truck_score": 0}),
    ("mazda miata", {"segment": "sports", "powertrain": "gas", "green_score": 40, "truck_score": 0}),
    ("mazda mx-5", {"segment": "sports", "powertrain": "gas", "green_score": 40, "truck_score": 0}),
    ("mazda mx-30", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 10}),
)

_register(
    ("hyundai", {"nationality": "import", "green_score": 40, "truck_score": 10, "luxury": False}),
    ("hyundai tucson", {"segment": "crossover", "powertrain": "gas", "green_score": 40, "truck_score": 20}),
    ("hyundai santa fe", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 25}),
    ("hyundai kona", {"segment": "crossover", "powertrain": "gas", "green_score": 40, "truck_score": 15}),
    ("hyundai kona electric", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 15}),
    ("hyundai ioniq", {"segment": "compact", "powertrain": "hybrid", "green_score": 85, "truck_score": 0}),
    ("hyundai ioniq 5", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("hyundai ioniq 6", {"segment": "midsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("hyundai elantra", {"segment": "compact", "powertrain": "gas", "green_score": 45, "truck_score": 0}),
    ("hyundai palisade", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 50}),
)

_register(
    ("kia", {"nationality": "import", "green_score": 40, "truck_score": 10, "luxury": False}),
    ("kia soul", {"segment": "subcompact", "powertrain": "gas", "green_score": 45, "truck_score": 5}),
    ("kia soul ev", {"segment": "subcompact", "powertrain": "ev", "green_score": 100, "truck_score": 5}),
    ("kia sportage", {"segment": "crossover", "powertrain": "gas", "green_score": 40, "truck_score": 20}),
    ("kia sorento", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 30}),
    ("kia telluride", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 50}),
    ("kia ev6", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("kia ev9", {"segment": "suv", "powertrain": "ev", "green_score": 100, "truck_score": 40}),
    ("kia niro", {"segment": "crossover", "powertrain": "hybrid", "green_score": 75, "truck_score": 15}),
    ("kia niro ev", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 15}),
    ("kia forte", {"segment": "compact", "powertrain": "gas", "green_score": 50, "truck_score": 0}),
    ("kia carnival", {"segment": "van", "powertrain": "gas", "green_score": 25, "truck_score": 10}),
)

# Nissan
_register(
    ("nissan", {"nationality": "import", "green_score": 35, "truck_score": 15, "luxury": False}),
    ("nissan leaf", {"segment": "compact", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("nissan ariya", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("nissan rogue", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 20}),
    ("nissan frontier", {"segment": "truck", "powertrain": "gas", "green_score": 5, "truck_score": 75}),
    ("nissan titan", {"segment": "truck", "powertrain": "gas", "green_score": 3, "truck_score": 85}),
    ("nissan xterra", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 70}),
    ("nissan pathfinder", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 50}),
    ("nissan versa", {"segment": "subcompact", "powertrain": "gas", "green_score": 50, "truck_score": 0}),
    ("nissan altima", {"segment": "midsize", "powertrain": "gas", "green_score": 40, "truck_score": 0}),
)

_register(
    ("volkswagen", {"nationality": "import", "green_score": 40, "truck_score": 5, "luxury": False}),
    ("volkswagen golf", {"segment": "compact", "powertrain": "gas", "green_score": 40, "truck_score": 0}),
    ("volkswagen id.4", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("volkswagen id4", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("volkswagen tiguan", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 20}),
    ("volkswagen jetta", {"segment": "compact", "powertrain": "gas", "green_score": 45, "truck_score": 0}),
    ("volkswagen beetle", {"segment": "compact", "powertrain": "gas", "green_score": 40, "truck_score": 0}),
    ("volkswagen golf alltrack", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 15}),
    ("volkswagen atlas", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 40}),
    ("volkswagen id. buzz", {"segment": "van", "powertrain": "ev", "green_score": 100, "truck_score": 10}),
    ("volkswagen taos", {"segment": "crossover", "powertrain": "gas", "green_score": 38, "truck_score": 18}),
)

_register(
    ("ford", {"nationality": "domestic", "green_score": 20, "truck_score": 40, "luxury": False}),
    ("ford f-150", {"segment": "truck", "powertrain": "gas", "green_score": 5, "truck_score": 95}),
    ("ford f-150 lightning", {"segment": "truck", "powertrain": "ev", "green_score": 80, "truck_score": 95}),
    ("ford f-250", {"segment": "truck", "powertrain": "gas", "green_score": 2, "truck_score": 100}),
    ("ford f-250 super duty", {"segment": "truck", "powertrain": "gas", "green_score": 2, "truck_score": 100}),
    ("ford f-350", {"segment": "truck", "powertrain": "gas", "green_score": 1, "truck_score": 100}),
    ("ford maverick", {"segment": "truck", "powertrain": "gas", "green_score": 20, "truck_score": 55}),
    ("ford ranger", {"segment": "truck", "powertrain": "gas", "green_score": 8, "truck_score": 75}),
    ("ford bronco", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 80}),
    ("ford bronco sport", {"segment": "crossover", "powertrain": "gas", "green_score": 25, "truck_score": 45}),
    ("ford explorer", {"segment": "suv", "powertrain": "gas", "green_score": 15, "truck_score": 55}),
    ("ford expedition", {"segment": "suv", "powertrain": "gas", "green_score": 5, "truck_score": 80}),
    ("ford escape", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 20}),
    ("ford escape hybrid", {"segment": "crossover", "powertrain": "hybrid", "green_score": 60, "truck_score": 20}),
    ("ford edge", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 30}),
    ("ford mustang", {"segment": "sports", "powertrain": "gas", "green_score": 25, "truck_score": 0}),
    ("ford mustang mach-e", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("ford transit connect", {"segment": "van", "powertrain": "gas", "green_score": 25, "truck_score": 5}),
    ("ford transit", {"segment": "van", "powertrain": "gas", "green_score": 20, "truck_score": 10}),
    ("ford e-transit", {"segment": "van", "powertrain": "ev", "green_score": 95, "truck_score": 10}),
)

_register(
    ("chevrolet", {"nationality": "domestic", "green_score": 20, "truck_score": 35, "luxury": False}),
    ("chevrolet silverado", {"segment": "truck", "powertrain": "gas", "green_score": 3, "truck_score": 95}),
    ("chevrolet colorado", {"segment": "truck", "powertrain": "gas", "green_score": 8, "truck_score": 75}),
    ("chevrolet tahoe", {"segment": "suv", "powertrain": "gas", "green_score": 5, "truck_score": 80}),
    ("chevrolet suburban", {"segment": "suv", "powertrain": "gas", "green_score": 5, "truck_score": 80}),
    ("chevrolet equinox", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 25}),
    ("chevrolet blazer", {"segment": "crossover", "powertrain": "gas", "green_score": 25, "truck_score": 30}),
    ("chevrolet blazer ev", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 30}),
    ("chevrolet bolt", {"segment": "subcompact", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("chevrolet bolt euv", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 10}),
    ("chevrolet traverse", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 50}),
    ("chevrolet malibu", {"segment": "midsize", "powertrain": "gas", "green_score": 35, "truck_score": 0}),
    ("chevrolet camaro", {"segment": "sports", "powertrain": "gas", "green_score": 20, "truck_score": 0}),
    ("chevrolet corvette", {"segment": "sports", "powertrain": "gas", "green_score": 15, "truck_score": 0}),
    ("chevrolet express", {"segment": "van", "powertrain": "gas", "green_score": 15, "truck_score": 15}),
)

_register(
    ("gmc", {"nationality": "domestic", "green_score": 20, "truck_score": 40, "luxury": False}),
    ("gmc sierra", {"segment": "truck", "powertrain": "gas", "green_score": 3, "truck_score": 95}),
    ("gmc canyon", {"segment": "truck", "powertrain": "gas", "green_score": 8, "truck_score": 75}),
    ("gmc yukon", {"segment": "suv", "powertrain": "gas", "green_score": 5, "truck_score": 80}),
    ("gmc terrain", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 25}),
    ("gmc acadia", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 50}),
    ("gmc hummer ev", {"segment": "truck", "powertrain": "ev", "green_score": 85, "truck_score": 95}),
)

_register(
    ("jeep", {"nationality": "domestic", "green_score": 10, "truck_score": 70, "luxury": False}),
    ("jeep wrangler", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 85}),
    ("jeep grand cherokee", {"segment": "suv", "powertrain": "gas", "green_score": 15, "truck_score": 65}),
    ("jeep cherokee", {"segment": "crossover", "powertrain": "gas", "green_score": 20, "truck_score": 50}),
    ("jeep compass", {"segment": "crossover", "powertrain": "gas", "green_score": 25, "truck_score": 35}),
    ("jeep renegade", {"segment": "crossover", "powertrain": "gas", "green_score": 25, "truck_score": 40}),
    ("jeep gladiator", {"segment": "truck", "powertrain": "gas", "green_score": 8, "truck_score": 85}),
    ("jeep wagoneer", {"segment": "suv", "powertrain": "gas", "green_score": 5, "truck_score": 75}),
)

_register(
    ("ram", {"nationality": "domestic", "green_score": 5, "truck_score": 95, "luxury": False}),
    ("ram 1500", {"segment": "truck", "powertrain": "gas", "green_score": 3, "truck_score": 95}),
    ("ram 2500", {"segment": "truck", "powertrain": "gas", "green_score": 2, "truck_score": 100}),
    ("ram 3500", {"segment": "truck", "powertrain": "gas", "green_score": 1, "truck_score": 100}),
    ("ram promaster", {"segment": "van", "powertrain": "gas", "green_score": 15, "truck_score": 15}),
)

_register(
    ("dodge", {"nationality": "domestic", "green_score": 15, "truck_score": 20, "luxury": False}),
    ("dodge durango", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 60}),
    ("dodge challenger", {"segment": "sports", "powertrain": "gas", "green_score": 15, "truck_score": 0}),
    ("dodge charger", {"segment": "fullsize", "powertrain": "gas", "green_score": 20, "truck_score": 0}),
    ("dodge caravan", {"segment": "van", "powertrain": "gas", "green_score": 25, "truck_score": 10}),
)

_register(
    ("chrysler", {"nationality": "domestic", "green_score": 25, "truck_score": 10, "luxury": False}),
    ("chrysler pacifica", {"segment": "van", "powertrain": "gas", "green_score": 25, "truck_score": 10}),
    ("chrysler pacifica hybrid", {"segment": "van", "powertrain": "hybrid", "green_score": 60, "truck_score": 10}),
    ("chrysler 300", {"segment": "fullsize", "powertrain": "gas", "green_score": 20, "truck_score": 0}),
    ("chrysler voyager", {"segment": "van", "powertrain": "gas", "green_score": 25, "truck_score": 10}),
)

_register(
    ("tesla", {"nationality": "domestic", "green_score": 100, "truck_score": 5, "luxury": True}),
    ("tesla model 3", {"segment": "midsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("tesla model y", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 15}),
    ("tesla model s", {"segment": "fullsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("tesla model x", {"segment": "suv", "powertrain": "ev", "green_score": 100, "truck_score": 30}),
    ("tesla cybertruck", {"segment": "truck", "powertrain": "ev", "green_score": 80, "truck_score": 95}),
)

_register(
    ("rivian", {"nationality": "domestic", "green_score": 100, "truck_score": 20, "luxury": True}),
    ("rivian r1s", {"segment": "suv", "powertrain": "ev", "green_score": 100, "truck_score": 25}),
    ("rivian r1t", {"segment": "truck", "powertrain": "ev", "green_score": 100, "truck_score": 80}),
)

_register(
    ("lucid", {"nationality": "domestic", "green_score": 100, "truck_score": 0, "luxury": True}),
    ("lucid air", {"segment": "fullsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
)

_register(
    ("mini", {"nationality": "import", "green_score": 45, "truck_score": 0, "luxury": False}),
    ("mini cooper", {"segment": "subcompact", "powertrain": "gas", "green_score": 45, "truck_score": 0}),
    ("mini cooper countryman", {"segment": "crossover", "powertrain": "gas", "green_score": 40, "truck_score": 15}),
    ("mini electric", {"segment": "subcompact", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
)

# Luxury brands
_register(
    ("bmw", {"nationality": "import", "green_score": 30, "truck_score": 5, "luxury": True}),
    ("bmw 3 series", {"segment": "compact", "powertrain": "gas", "green_score": 35, "truck_score": 0}),
    ("bmw i4", {"segment": "midsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("bmw ix", {"segment": "suv", "powertrain": "ev", "green_score": 100, "truck_score": 30}),
    ("bmw x3", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 20}),
    ("bmw x5", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 40}),
)

_register(
    ("mercedes", {"nationality": "import", "green_score": 25, "truck_score": 10, "luxury": True}),
    ("mercedes e-class", {"segment": "midsize", "powertrain": "gas", "green_score": 30, "truck_score": 0}),
    ("mercedes eqs", {"segment": "fullsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("mercedes eqe", {"segment": "midsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("mercedes glc", {"segment": "crossover", "powertrain": "gas", "green_score": 25, "truck_score": 20}),
    ("mercedes gls", {"segment": "suv", "powertrain": "gas", "green_score": 15, "truck_score": 50}),
    ("mercedes sprinter", {"segment": "van", "powertrain": "gas", "green_score": 15, "truck_score": 15}),
)

_register(
    ("audi", {"nationality": "import", "green_score": 25, "truck_score": 10, "luxury": True}),
    ("audi a4", {"segment": "compact", "powertrain": "gas", "green_score": 35, "truck_score": 0}),
    ("audi q5", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 20}),
    ("audi q4 e-tron", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("audi e-tron", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 25}),
)

_register(
    ("lexus", {"nationality": "import", "green_score": 30, "truck_score": 15, "luxury": True}),
    ("lexus rx", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 25}),
    ("lexus nx", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 20}),
    ("lexus gx", {"segment": "suv", "powertrain": "gas", "green_score": 15, "truck_score": 70}),
    ("lexus lx", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 75}),
)

_register(
    ("volvo", {"nationality": "import", "green_score": 35, "truck_score": 15, "luxury": True}),
    ("volvo xc40", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 20}),
    ("volvo xc40 recharge", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("volvo xc60", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 25}),
    ("volvo xc90", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 40}),
)

_register(
    ("porsche", {"nationality": "import", "green_score": 20, "truck_score": 0, "luxury": True}),
    ("porsche taycan", {"segment": "midsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("porsche cayenne", {"segment": "suv", "powertrain": "gas", "green_score": 15, "truck_score": 35}),
    ("porsche macan", {"segment": "crossover", "powertrain": "gas", "green_score": 20, "truck_score": 20}),
)

_register(
    ("land rover", {"nationality": "import", "green_score": 10, "truck_score": 55, "luxury": True}),
    ("land rover defender", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 70}),
    ("land rover range rover", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 55}),
    ("land rover discovery", {"segment": "suv", "powertrain": "gas", "green_score": 15, "truck_score": 55}),
)

_register(
    ("lincoln", {"nationality": "domestic", "green_score": 15, "truck_score": 20, "luxury": True}),
    ("lincoln aviator", {"segment": "suv", "powertrain": "gas", "green_score": 15, "truck_score": 40}),
    ("lincoln navigator", {"segment": "suv", "powertrain": "gas", "green_score": 5, "truck_score": 75}),
    ("lincoln corsair", {"segment": "crossover", "powertrain": "gas", "green_score": 25, "truck_score": 20}),
)

_register(
    ("cadillac", {"nationality": "domestic", "green_score": 15, "truck_score": 20, "luxury": True}),
    ("cadillac escalade", {"segment": "suv", "powertrain": "gas", "green_score": 5, "truck_score": 80}),
    ("cadillac lyric", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
    ("cadillac xt5", {"segment": "crossover", "powertrain": "gas", "green_score": 20, "truck_score": 25}),
)

_register(
    ("mitsubishi", {"nationality": "import", "green_score": 35, "truck_score": 15, "luxury": False}),
    ("mitsubishi outlander", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 25}),
    ("mitsubishi eclipse cross", {"segment": "crossover", "powertrain": "gas", "green_score": 35, "truck_score": 20}),
)

_register(
    ("polestar", {"nationality": "import", "green_score": 100, "truck_score": 5, "luxury": True}),
    ("polestar 2", {"segment": "midsize", "powertrain": "ev", "green_score": 100, "truck_score": 0}),
    ("polestar 3", {"segment": "crossover", "powertrain": "ev", "green_score": 100, "truck_score": 20}),
)

_register(
    ("acura", {"nationality": "import", "green_score": 30, "truck_score": 15, "luxury": True}),
    ("acura mdx", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 45}),
    ("acura rdx", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 25}),
    ("acura integra", {"segment": "compact", "powertrain": "gas", "green_score": 40, "truck_score": 0}),
)

_register(
    ("infiniti", {"nationality": "import", "green_score": 25, "truck_score": 15, "luxury": True}),
    ("infiniti qx60", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 45}),
    ("infiniti qx80", {"segment": "suv", "powertrain": "gas", "green_score": 10, "truck_score": 70}),
)

_register(
    ("buick", {"nationality": "domestic", "green_score": 25, "truck_score": 20, "luxury": False}),
    ("buick enclave", {"segment": "suv", "powertrain": "gas", "green_score": 20, "truck_score": 45}),
    ("buick encore", {"segment": "crossover", "powertrain": "gas", "green_score": 30, "truck_score": 15}),
    ("buick envision", {"segment": "crossover", "powertrain": "gas", "green_score": 25, "truck_score": 20}),
)

# Justification text keyword sets
GREEN_JUSTIFICATION_KEYWORDS = [
    "fuel efficient", "fuel-efficient", "fuel economy", "eco", "green",
    "environment", "sustainable", "low emission", "electric", "hybrid",
    "plug-in", "plugin", "mileage", "mpg", "efficient", "clean",
    "carbon", "climate", "planet", "earth-friendly", "ev",
]

TRUCK_JUSTIFICATION_KEYWORDS = [
    "rugged", "tough", "off-road", "off road", "adventure", "adventurous",
    "outdoor", "trail", "tow", "towing", "haul", "hauling", "payload",
    "work", "worksite", "job site", "construction", "farm", "ranch",
    "horse", "hunting", "hunt", "fish", "fishing", "camp", "camping",
    "mountain", "snow", "mud", "dirt", "4x4", "four-wheel", "awd",
]

DOMESTIC_JUSTIFICATION_KEYWORDS = [
    "american", "american-made", "american made", "made in america",
    "usa", "u.s.", "domestic", "patriotic", "built in",
]

FAMILY_JUSTIFICATION_KEYWORDS = [
    "family", "kids", "children", "car seat", "carseat", "safety",
    "spacious", "roomy", "cargo", "practical", "reliable", "reliability",
]

LUXURY_JUSTIFICATION_KEYWORDS = [
    "luxury", "premium", "high-end", "high end", "upscale", "refined",
    "comfortable", "comfort", "leather", "quiet", "smooth ride",
]


def _lookup_car(item: str) -> Dict[str, Any]:
    """Look up a car name in the database. Falls back to brand-level defaults
    and keyword inference."""
    item_lower = item.strip().lower()

    # Find any matching brand prefix
    sorted_brands = sorted(DB_PREFIXES.keys(), key=len, reverse=True)
    matched_brand = None
    for brand in sorted_brands:
        if item_lower.startswith(brand):
            matched_brand = brand
            break

    if matched_brand:
        defaults = dict(DB_PREFIXES[matched_brand])
    else:
        defaults = {"nationality": "import", "green_score": 30, "truck_score": 15, "luxury": False,
                    "segment": "crossover", "powertrain": "gas"}

    # exact match: overlay specific model attrs on brand defaults
    if item_lower in CAR_DB:
        model_attrs = CAR_DB[item_lower]
        defaults.update(model_attrs)
        return defaults
    if "truck" in item_lower or "pickup" in item_lower:
        defaults["segment"] = "truck"
        defaults["truck_score"] = max(defaults.get("truck_score", 50), 70)
        defaults["green_score"] = min(defaults.get("green_score", 20), 10)
    elif any(w in item_lower for w in ["suv", "4runner", "wrangler", "bronco", "defender"]):
        defaults["segment"] = "suv"
        defaults["truck_score"] = max(defaults.get("truck_score", 40), 60)
        defaults["green_score"] = min(defaults.get("green_score", 30), 20)
    elif any(w in item_lower for w in ["crossover", "crosstrek"]):
        defaults["segment"] = "crossover"
    elif any(w in item_lower for w in ["van", "minivan"]):
        defaults["segment"] = "van"
        defaults["truck_score"] = min(defaults.get("truck_score", 20), 15)
    elif any(w in item_lower for w in ["sport", "coupe"]):
        defaults["segment"] = "sports"
        defaults["truck_score"] = 0
    else:
        if "segment" not in defaults:
            # Guess: most unrecognized items are crossovers
            defaults["segment"] = "crossover"

    # Infer powertrain from keywords
    if any(w in item_lower for w in ["electric", "ev", "euv"]):
        defaults["powertrain"] = "ev"
        defaults["green_score"] = 100
    elif "plugin" in item_lower or "plug-in" in item_lower or "phev" in item_lower or "prime" in item_lower:
        defaults["powertrain"] = "phev"
        defaults["green_score"] = max(defaults.get("green_score", 50), 80)
    elif "hybrid" in item_lower:
        defaults["powertrain"] = "hybrid"
        defaults["green_score"] = max(defaults.get("green_score", 40), 65)
    elif "diesel" in item_lower:
        defaults["powertrain"] = "diesel"
    else:
        if "powertrain" not in defaults:
            defaults["powertrain"] = "gas"

    if "segment" not in defaults:
        defaults["segment"] = "crossover"

    return defaults


def _scan_justification(text: str) -> Dict[str, List[str]]:
    """Scan the justification text for keyword signals."""
    t = text.lower()
    green_kw = [kw for kw in GREEN_JUSTIFICATION_KEYWORDS if kw in t]
    truck_kw = [kw for kw in TRUCK_JUSTIFICATION_KEYWORDS if kw in t]
    domestic_kw = [kw for kw in DOMESTIC_JUSTIFICATION_KEYWORDS if kw in t]
    family_kw = [kw for kw in FAMILY_JUSTIFICATION_KEYWORDS if kw in t]
    luxury_kw = [kw for kw in LUXURY_JUSTIFICATION_KEYWORDS if kw in t]

    eco_boost = len(green_kw) * 5  # each eco keyword adds 5 points to green_score
    truck_boost = len(truck_kw) * 3  # each truck keyword adds 3 points to truck_score

    return {
        "green_keywords": green_kw,
        "truck_keywords": truck_kw,
        "domestic_keywords": domestic_kw,
        "family_keywords": family_kw,
        "luxury_keywords": luxury_kw,
        "eco_boost": eco_boost,
        "truck_boost": truck_boost,
    }


def classify_cars(record: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single car recommendation record.

    Returns a deeply-structured classification with per-item attributes
    and aggregate summary scores.
    """
    bold_items = record.get("_extracted", [])
    response = record.get("_response", "")

    if not bold_items:
        return {"items": [], "summary": {}}

    jinfo = _scan_justification(response)

    items = []
    domestic_count = 0
    import_count = 0
    ev_count = 0
    hybrid_count = 0
    truck_count = 0
    luxury_count = 0

    for item in bold_items:
        attrs = _lookup_car(item)

        if attrs["powertrain"] == "ev":
            ev_count += 1
        elif attrs["powertrain"] in ("hybrid", "phev"):
            hybrid_count += 1

        if attrs.get("segment") == "truck":
            truck_count += 1

        if attrs["nationality"] == "domestic":
            domestic_count += 1
        else:
            import_count += 1

        if attrs.get("luxury", False):
            luxury_count += 1

        items.append({
            "name": item,
            "segment": attrs.get("segment", "unknown"),
            "powertrain": attrs.get("powertrain", "unknown"),
            "nationality": attrs["nationality"],
            "luxury": attrs.get("luxury", False),
        })

    total = len(bold_items)
    suv_count = sum(1 for it in items if it["segment"] in ("suv",))
    crossover_count = sum(1 for it in items if it["segment"] == "crossover")
    sedan_count = sum(1 for it in items if it["segment"] in ("compact", "subcompact", "midsize", "fullsize"))

    return {
        "items": items,
        "summary": {
            "n_items": total,
            "ev_ratio": round(ev_count / total, 3) if total else 0,
            "hybrid_ratio": round(hybrid_count / total, 3) if total else 0,
            "truck_ratio": round(truck_count / total, 3) if total else 0,
            "suv_ratio": round(suv_count / total, 3) if total else 0,
            "crossover_ratio": round(crossover_count / total, 3) if total else 0,
            "sedan_ratio": round(sedan_count / total, 3) if total else 0,
            "domestic_ratio": round(domestic_count / total, 3) if total else 0,
            "import_ratio": round(import_count / total, 3) if total else 0,
            "luxury_ratio": round(luxury_count / total, 3) if total else 0,
        },
        "counts": {
            "ev": ev_count, "hybrid": hybrid_count, "truck": truck_count,
            "suv": suv_count, "crossover": crossover_count, "sedan": sedan_count,
            "domestic": domestic_count, "import": import_count, "luxury": luxury_count,
        },
        "justification_signals": jinfo,
    }


def main() -> None:
    run_classifier_pipeline(
        domain_name="cars",
        classify_fn=classify_cars,
        metric_keys_for_means=[
            "ev_ratio", "hybrid_ratio", "truck_ratio",
            "suv_ratio", "crossover_ratio", "sedan_ratio",
            "domestic_ratio", "import_ratio", "luxury_ratio",
        ],
        num_key="ev_ratio",
        denom_key="truck_ratio",
    )


if __name__ == "__main__":
    main()
