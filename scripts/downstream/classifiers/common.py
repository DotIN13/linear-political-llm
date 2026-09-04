"""Shared utilities for all domain classifiers."""

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

GROUP_ORDER = ["baseline",
               "dem_impl_no_img", "dem_impl_img",
               "rep_impl_no_img", "rep_impl_img",
               "dem_explicit", "rep_explicit",
               "dem_img_only", "rep_img_only"]

GROUP_DISPLAY = {
    "baseline": "Baseline",
    "dem_impl_no_img": "Dem(impl,no-img)",
    "dem_impl_img": "Dem(impl,img)",
    "rep_impl_no_img": "Rep(impl,no-img)",
    "rep_impl_img": "Rep(impl,img)",
    "dem_explicit": "Dem(explicit)",
    "rep_explicit": "Rep(explicit)",
    "dem_img_only": "Dem(img-only)",
    "rep_img_only": "Rep(img-only)",
    "dem_profile": "Dem(profile)",
    "rep_profile": "Rep(profile)",
    "dem_impl": "Dem(implicit)",
    "rep_impl": "Rep(implicit)",
}


def _detect_group_order(records: list) -> List[str]:
    """Detect the actual group order from records, preserving insertion order."""
    seen = []
    for r in records:
        g = r.get("group", "")
        if g and g not in seen:
            seen.append(g)
    return seen if seen else GROUP_ORDER


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL recommendation file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_records(records: List[Dict[str, Any]], path: str) -> None:
    """Save enriched records as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def group_by_condition(
    records: List[Dict[str, Any]],
    group_order: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by condition. Auto-detects group order if not provided."""
    if group_order is None:
        group_order = _detect_group_order(records)
    grouped: Dict[str, List[Dict]] = {g: [] for g in group_order}
    for r in records:
        g = r.get("group", "neutral")
        if g in grouped:
            grouped[g].append(r)
    return grouped


def aggregate_counts(
    grouped: Dict[str, List[Dict[str, Any]]],
    cls_key: str = "_classification_v2",
) -> Dict[str, Dict[str, int]]:
    """Sum classification counts per group."""
    aggs = {}
    for g, records in grouped.items():
        agg: Dict[str, int] = defaultdict(int)
        for r in records:
            cls = r.get(cls_key, {})
            if isinstance(cls, dict):
                # new deep structure: {"items": [...], "summary": {...}}
                summary = cls.get("summary", cls)
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        agg[k] += v
        aggs[g] = dict(agg)
    return aggs


def aggregate_means(
    grouped: Dict[str, List[Dict[str, Any]]],
    cls_key: str = "_classification_v2",
) -> Dict[str, Dict[str, float]]:
    """Compute per-persona means of scalar classification values."""
    aggs: Dict[str, Dict[str, List[float]]] = {}
    for g, records in grouped.items():
        aggs[g] = defaultdict(list)
        for r in records:
            cls = r.get(cls_key, {})
            summary = cls.get("summary", cls)
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    aggs[g][k].append(float(v))
    means = {}
    for g, recs in grouped.items():
        means[g] = {k: sum(vals) / len(vals) if vals else 0.0
                    for k, vals in aggs[g].items()}
    return means


def print_count_table(
    grouped: Dict[str, List[Dict[str, Any]]],
    metric_keys: List[str],
    cls_key: str = "_classification_v2",
) -> None:
    """Print a table of total counts per group per metric."""
    aggs = aggregate_counts(grouped, cls_key)
    group_order = list(grouped.keys())

    header = f"  {'Metric':<22s}"
    for g in group_order:
        header += f" {GROUP_DISPLAY.get(g, g):>16s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mk in metric_keys:
        row = f"  {mk:<22s}"
        for g in group_order:
            row += f" {aggs.get(g, {}).get(mk, 0):>16d}"
        print(row)


def print_mean_table(
    grouped: Dict[str, List[Dict[str, Any]]],
    metric_keys: List[str],
    cls_key: str = "_classification_v2",
) -> None:
    """Print a table of per-persona means per group."""
    means = aggregate_means(grouped, cls_key)
    group_order = list(grouped.keys())

    header = f"  {'Metric':<22s}"
    for g in group_order:
        header += f" {GROUP_DISPLAY.get(g, g):>16s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mk in metric_keys:
        row = f"  {mk:<22s}"
        for g in group_order:
            val = means.get(g, {}).get(mk, 0.0)
            row += f" {val:>16.1f}"
        print(row)


def print_ratios(
    grouped: Dict[str, List[Dict[str, Any]]],
    num_key: str,
    denom_key: str,
    cls_key: str = "_classification_v2",
) -> None:
    """Print num/denom ratios per group."""
    aggs = aggregate_counts(grouped, cls_key)
    for g in grouped.keys():
        num = aggs.get(g, {}).get(num_key, 0)
        denom = aggs.get(g, {}).get(denom_key, 0)
        ratio = f"{num/denom:.2f}" if denom > 0 else ("inf" if num > 0 else "n/a")
        print(f"    {GROUP_DISPLAY.get(g, g):<20s} {num} / {denom} = {ratio}")


def extract_top_items(
    grouped: Dict[str, List[Dict[str, Any]]],
    extract_key: str = "_extracted",
    top_n: int = 10,
) -> Dict[str, List[Tuple[str, int]]]:
    """Get top-N most frequent bold items per group."""
    result = {}
    for g, records in grouped.items():
        counter: Counter = Counter()
        for r in records:
            items = r.get(extract_key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str):
                        counter[item.strip()] += 1
                    elif isinstance(item, dict):
                        name = item.get("matched_name") or item.get("city", "")
                        counter[name.strip()] += 1
        result[g] = counter.most_common(top_n)
    return result


def print_top_items(
    grouped: Dict[str, List[Dict[str, Any]]],
    extract_key: str = "_extracted",
    top_n: int = 10,
) -> None:
    """Print top extracted items per group."""
    top = extract_top_items(grouped, extract_key, top_n)
    for g in grouped.keys():
        items_str = ", ".join(f"{it}({c})" for it, c in top[g]) if top[g] else "(none)"
        print(f"    {GROUP_DISPLAY.get(g, g):<20s} {items_str}")


def print_sample_responses(
    grouped: Dict[str, List[Dict[str, Any]]],
    max_chars: int = 300,
) -> None:
    """Print one sample response per group."""
    for g, recs in grouped.items():
        for r in recs:
            resp = r.get("_response", "")
            if resp and not resp.startswith("[ERROR"):
                print(f"\n    [{GROUP_DISPLAY.get(g, g)}] {r['persona_id']}")
                print(f"    Objects: {', '.join(r.get('objects', [])[:4])}...")
                print(f"    Response: {resp[:max_chars]}...")
                break


def run_classifier_pipeline(
    domain_name: str,
    classify_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    metric_keys_for_counts: Optional[List[str]] = None,
    metric_keys_for_means: Optional[List[str]] = None,
    num_key: Optional[str] = None,
    denom_key: Optional[str] = None,
    cls_key: str = "_classification_v2",
    file_prefix: str = "lvis_",
) -> None:
    """Standard pipeline: load, classify, save, summarize.

    Args:
        domain_name: 'travel', 'cars', etc.
        classify_fn: function(record) -> classification dict
        metric_keys_for_counts: keys for the count summary table
        metric_keys_for_means: keys for the means summary table
        num_key, denom_key: keys for the ratio line
        cls_key: key to store classification under in each record
        file_prefix: prefix for input/output filenames (default 'lvis_')
                     e.g. 'pol_to_apol_' → pol_to_apol_travel_recommendations.jsonl
    """
    data_dir = os.path.join(ROOT_DIR, "data", "lvis_persona")
    input_path = os.path.join(data_dir, f"{file_prefix}{domain_name}_recommendations.jsonl")
    output_path = os.path.join(data_dir, f"{file_prefix}{domain_name}_classified.jsonl")

    if not os.path.exists(input_path):
        print(f"SKIP {domain_name}: input not found at {input_path}")
        return

    records = load_records(input_path)
    print(f"\nLoaded {len(records)} records from {input_path}")

    for r in records:
        r[cls_key] = classify_fn(r)

    save_records(records, output_path)
    print(f"Saved classified records to {output_path}")

    grouped = group_by_condition(records)

    print(f"\n{'─'*70}")
    print(f"  {domain_name.upper()} CLASSIFICATION SUMMARY")
    print(f"{'─'*70}")

    if metric_keys_for_counts:
        print(f"\n  --- Total counts per group ---")
        print_count_table(grouped, metric_keys_for_counts, cls_key)

    if metric_keys_for_means:
        print(f"\n  --- Per-persona means ---")
        print_mean_table(grouped, metric_keys_for_means, cls_key)

    if num_key and denom_key:
        print(f"\n  --- {num_key} : {denom_key} ratio ---")
        print_ratios(grouped, num_key, denom_key, cls_key)

    print(f"\n  --- Top recommendations ---")
    print_top_items(grouped, extract_key="_extracted")

    print(f"\n  --- Sample responses ---")
    print_sample_responses(grouped)

    print(f"\n{'─'*70}\n")
