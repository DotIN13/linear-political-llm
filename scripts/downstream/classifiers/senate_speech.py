#!/usr/bin/env python3
"""
Classifier for Senate speech evaluation experiment (apol→pol).

Parses LLM judge evaluations (structured JSON) from speech outlines and
computes per-group mean political lean scores across 5 dimensions.

Usage: python -m scripts.downstream.classifiers.senate_speech
"""

import json
import os
import sys
from typing import Any, Dict, List

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

DIMENSIONS = [
    "overall_lean",
    "economic_lean",
    "social_lean",
    "foreign_policy_lean",
    "institutional_trust",
]

DIM_LABELS = {
    "overall_lean": "Overall lean (-1=Rep / +1=Dem)",
    "economic_lean": "Economic (-1=free mkt / +1=regulation)",
    "social_lean": "Social (-1=traditional / +1=progressive)",
    "foreign_policy_lean": "Foreign (-1=hard power / +1=diplomacy)",
    "institutional_trust": "Instit. trust (-1=skeptical / +1=trusts)",
}


def parse_speech_eval(record: Dict[str, Any]) -> Dict[str, Any]:
    eval_data = record.get("_judge_eval", {})
    if not isinstance(eval_data, dict):
        return {"parse_error": True, "dims": {}}

    if "error" in eval_data:
        return {"parse_error": True, "error_msg": str(eval_data["error"]), "dims": {}}

    dims = {}
    for dim in DIMENSIONS:
        if dim in eval_data:
            try:
                dims[dim] = float(eval_data[dim])
            except (ValueError, TypeError):
                dims[dim] = None

    return {
        "parse_error": False,
        "dims": dims,
        "rationale": eval_data.get("rationale", ""),
    }


def run_senate_speech_pipeline() -> None:
    data_dir = os.path.join(ROOT_DIR, "data", "lvis_persona")
    input_path = os.path.join(data_dir, "apol_to_pol_senate_speech.jsonl")
    output_path = os.path.join(data_dir, "apol_to_pol_senate_speech_classified.jsonl")

    if not os.path.exists(input_path):
        print(f"SKIP: input not found at {input_path}")
        return

    records = load_records(input_path)
    print(f"\n{'='*60}")
    print(f"  SENATE SPEECH EVALUATION")
    print(f"{'='*60}")
    print(f"Loaded {len(records)} records")

    for r in records:
        r["_speech_classification"] = parse_speech_eval(r)

    save_records(records, output_path)
    print(f"Saved classified records to {output_path}")

    grouped: Dict[str, List[Dict]] = {g: [] for g in GROUP_ORDER_ATP}
    for r in records:
        g = r.get("group", "")
        if g in grouped:
            grouped[g].append(r)

    # Per-group means and standard errors
    print(f"\n  --- Speech lean (mean ± sem) ---")
    header = f"  {'Dimension':<22s}"
    for g in GROUP_ORDER_ATP:
        header += f" {GROUP_DISPLAY_ATP[g]:>24s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dim in DIMENSIONS:
        row = f"  {dim:<22s}"
        for g in GROUP_ORDER_ATP:
            vals = []
            for r in grouped[g]:
                cls = r.get("_speech_classification", {})
                dims = cls.get("dims", {})
                if dim in dims and dims[dim] is not None:
                    vals.append(dims[dim])
            if vals:
                n = len(vals)
                mu = sum(vals) / n
                sem = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5 / (n ** 0.5)
                row += f" {mu:>12.4f} ± {sem:.4f}"
            else:
                row += f" {'n/a':>24s}"
        print(row)

    # Group difference (t-test-style)
    print(f"\n  --- Group differences (dem_impl - rep_impl) ---")
    for dim in DIMENSIONS:
        vals_dem = []
        vals_rep = []
        for r in grouped["dem_impl"]:
            cls = r.get("_speech_classification", {})
            dims = cls.get("dims", {})
            if dim in dims and dims[dim] is not None:
                vals_dem.append(dims[dim])
        for r in grouped["rep_impl"]:
            cls = r.get("_speech_classification", {})
            dims = cls.get("dims", {})
            if dim in dims and dims[dim] is not None:
                vals_rep.append(dims[dim])
        if vals_dem and vals_rep:
            mu_d = sum(vals_dem) / len(vals_dem)
            mu_r = sum(vals_rep) / len(vals_rep)
            diff = mu_d - mu_r
            # Pooled std
            vard = sum((v - mu_d) ** 2 for v in vals_dem) / len(vals_dem)
            varr = sum((v - mu_r) ** 2 for v in vals_rep) / len(vals_rep)
            se = (vard / len(vals_dem) + varr / len(vals_rep)) ** 0.5
            t_stat = diff / se if se > 0 else 0
            marker = " ***" if abs(t_stat) > 2.5 else (" **" if abs(t_stat) > 1.5 else "")
            print(f"    {dim:<22s} Δ={diff:+.4f}  t={t_stat:+.2f}  se={se:.4f}{marker}")

    # Distribution summary
    print(f"\n  --- Distribution summary (overall_lean) ---")
    for g in GROUP_ORDER_ATP:
        vals = []
        for r in grouped[g]:
            cls = r.get("_speech_classification", {})
            dims = cls.get("dims", {})
            if "overall_lean" in dims and dims["overall_lean"] is not None:
                vals.append(dims["overall_lean"])
        if vals:
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            p5 = vals_sorted[max(0, n // 20)]
            p25 = vals_sorted[n // 4]
            p50 = vals_sorted[n // 2]
            p75 = vals_sorted[3 * n // 4]
            p95 = vals_sorted[min(n - 1, n * 19 // 20)]
            dem_lean = sum(1 for v in vals if v > 0.2)
            rep_lean = sum(1 for v in vals if v < -0.2)
            neutral = n - dem_lean - rep_lean
            print(f"    {GROUP_DISPLAY_ATP[g]:<20s} "
                  f"n={n}  mean={sum(vals)/n:.3f}  "
                  f"[{p5:.2f} {p25:.2f} {p50:.2f} {p75:.2f} {p95:.2f}]  "
                  f"Dem:{dem_lean}  Neutral:{neutral}  Rep:{rep_lean}")

    # Sample
    print(f"\n  --- Sample speeches + evals ---")
    for g in GROUP_ORDER_ATP:
        for r in grouped[g]:
            sp = r.get("_speech", "")
            ev = r.get("_judge_eval", {})
            if sp and not sp.startswith("[ERROR") and isinstance(ev, dict):
                overall = ev.get("overall_lean", "?")
                rationale = ev.get("rationale", "")[:200]
                print(f"\n    [{GROUP_DISPLAY_ATP[g]}] {r['persona_id']} "
                      f"(lean={overall})")
                print(f"    Speech: {sp[:300]}...")
                print(f"    Rationale: {rationale}...")
                break
    print()


if __name__ == "__main__":
    run_senate_speech_pipeline()
