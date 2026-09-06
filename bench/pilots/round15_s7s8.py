"""Round 15: s7 (family-chat reply) and s8 (letter to a representative).

The two new tasks, run across the handles that matter and judged on one rubric.
The design is deliberately small in the number of things that move:

    task            s7, s8                     -- the thing being compared
    photo band      low / mid / high            -- the independent variable
    conversation    chat / agentic              -- the other live handle
    question        12 per task                 -- runs, not a factor
    persona         6 per band                  -- the sample
    rep             1..R                        -- the same cell, measured again

    plus a no-photo baseline per (task, question, conversation).

s7 and s8 share their twelve issues in the same order and are scored with the
same rubric (s2's, reused -- see the note on each surface), so a difference
between them is the task shape and not the subject or the scale.

`rep` exists because nothing before this round could measure its own
repeatability: `trial_key` dedups, so a second reading of an identical cell was
silently dropped as already-done. An identical condition once gave 5.7/5.0/4.3
and then 3.2/5.0/4.2 with nothing changed, which is why every single-reading
number in this project is provisional. Judge results cache on
(answer text, rubric fingerprint), so a repeat that reproduces its text costs a
GPU call and no judge call -- and **the cache-hit rate is itself the
repeatability measurement**.

Phases:
    plan     -- print the design and the counts, run nothing
    smoke    -- one trial per (task, conversation, band, condition)
    run      -- the whole plan
    export   -- join runs/round15/judges.jsonl and write the upload

The judge is the standard step and is not run from here:
    python -m bench.cli judge --run runs/round15
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from bench import registry
from bench.judges.specs import LEAN_MAP, judge_specs
from bench.store import git_rev, measurement_rev, trial_key
from bench.types import Item, Trial

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "runs", "round15")
TRIALS_PATH = os.path.join(OUT_DIR, "trials.jsonl")
JUDGES_PATH = os.path.join(OUT_DIR, "judges.jsonl")
UPLOADS = os.path.join(ROOT, "agent-bridge-tmp", "uploads", "round15")
ITEMS_FILE = os.path.join(ROOT, "items", "explore_bucket_v1.jsonl")

SURFACE_IDS = ("s7_family_chat", "s8_letter_answered")
SCHEMES = ("chat", "agentic")
BUCKETS = ("low", "mid", "high")
ITEMS_PER_BUCKET = 6
SEED = 20260906
DEFAULT_REPS = 2


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #
def load_items(path: str = ITEMS_FILE) -> List[Dict[str, Any]]:
    per_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "explore":
                continue
            bucket = row.get("bucket") or _bucket_of(row)
            row["bucket"] = bucket
            per_bucket[bucket].append(row)
    out: List[Dict[str, Any]] = []
    for bucket in BUCKETS:
        out.extend(per_bucket[bucket][:ITEMS_PER_BUCKET])
    return out


def _bucket_of(row: Dict[str, Any]) -> str:
    scores = row.get("image_scores") or []
    mean = sum(scores) / len(scores) if scores else 0.0
    return "low" if mean < -0.5 else ("high" if mean > 0.5 else "mid")


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def build_plan(items: Sequence[Dict[str, Any]], reps: int = DEFAULT_REPS) -> List[Dict[str, Any]]:
    registry.load_all()
    plan: List[Dict[str, Any]] = []
    for sid in SURFACE_IDS:
        surface = registry.get_surface(sid)()
        for scheme in SCHEMES:
            for qid in surface.question_ids():
                for rep in range(reps):
                    variant = {"scheme": scheme, "question": qid, "rep": rep}
                    for row in items:
                        item = Item.from_dict(row)
                        trial = surface.build(item, "C", dict(variant), seed=SEED)
                        plan.append(_entry(trial, sid, "C", scheme, qid, rep, row, item))
                    # The no-photo baseline is item-invariant: identical text for
                    # every persona, so it runs once per cell rather than 18 times.
                    blank = Item(item_id="no_image", images=[], image_paths=[],
                                 image_scores=[], stratum=-1)
                    trial = surface.build(blank, "E", dict(variant), seed=SEED)
                    plan.append(_entry(trial, sid, "E", scheme, qid, rep,
                                       {"bucket": "none"}, blank))
    return plan


def _entry(trial: Trial, sid: str, condition: str, scheme: str, qid: str, rep: int,
           row: Dict[str, Any], item: Item) -> Dict[str, Any]:
    ds = (trial.meta or {}).get("dataset") or {}
    return {"trial": trial, "surface": sid, "condition": condition, "scheme": scheme,
            "question": qid, "rep": rep, "bucket": row.get("bucket", "none"),
            "item_id": item.item_id, "image_mean": item.image_mean,
            "image_scores": list(item.image_scores),
            "domain": ds.get("domain"), "topic": ds.get("topic"), "seed": SEED}


def phase_plan(reps: int = DEFAULT_REPS) -> None:
    plan = build_plan(load_items(), reps)
    print(f"[plan] {len(plan)} trials, reps={reps}")
    for key, n in sorted(Counter(
            (e["surface"], e["scheme"], e["condition"]) for e in plan).items()):
        print(f"   {key[0]:<20} {key[1]:<9} {key[2]}  {n}")
    print(f"   per rep: {len(plan) // reps}")
    print(f"   judge calls at worst: {len(plan)} (fewer if repeats reproduce their text)")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _smoke_slice(plan: List[Dict[str, Any]], n: int = 0) -> List[Dict[str, Any]]:
    """One per (task, conversation, band, condition) -- n=0 means all of them.

    A prefix of the plan is one task and one conversation style, which is the
    wrong thing to smoke: the risk is the agentic transcript and s8's inserted
    exchange, and a prefix touches neither.
    """
    picked: List[Dict[str, Any]] = []
    seen = set()
    for entry in plan:
        key = (entry["surface"], entry["scheme"], entry["bucket"], entry["condition"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(entry)
        if n and len(picked) >= n:
            break
    return picked


def phase_run(adaptor: Any = None, limit: int = 0, smoke: bool = False,
              reps: int = DEFAULT_REPS) -> int:
    registry.load_all()
    plan = build_plan(load_items(), reps)
    if smoke:
        plan = _smoke_slice(plan, limit)
    elif limit:
        plan = plan[:limit]

    if adaptor is None:
        from bench.adaptors.vllm_server import VLLMServerAdaptor
        adaptor = VLLMServerAdaptor(
            model=os.environ.get("VLLM_SERVED_MODEL", "qwen3-vl-8b-instruct"),
            base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8073/v1"),
            seed=SEED,
        )
    adaptor.setup()

    rev = measurement_rev(ROOT, note=f"adaptor={adaptor.name}")
    code_rev = git_rev(ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(TRIALS_PATH):
        with open(TRIALS_PATH, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["trial_key"])

    surfaces = {sid: registry.get_surface(sid)() for sid in SURFACE_IDS}
    written = n_err = 0
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        for i, entry in enumerate(plan, 1):
            trial = entry["trial"]
            key = trial_key(entry["surface"], entry["item_id"], entry["condition"],
                            trial.variant, adaptor.name, adaptor.model, entry["seed"], rev)
            if key in done:
                continue
            resp = adaptor.run(trial)
            outcome = surfaces[entry["surface"]].extract(resp, trial)
            record = {
                "trial_key": key, "run_id": "round15",
                "code_rev": code_rev, "measurement_rev": rev,
                "adaptor": adaptor.name, "model": adaptor.model,
                "surface": entry["surface"], "surface_family": "generation",
                "condition": entry["condition"], "scheme": entry["scheme"],
                "question": entry["question"], "rep": entry["rep"],
                "domain": entry["domain"], "topic": entry["topic"],
                "bucket": entry["bucket"], "item_id": entry["item_id"],
                "seed": entry["seed"], "image_mean": entry["image_mean"],
                "image_scores": entry["image_scores"], "variant": trial.variant,
                "conversation_sha": trial.conversation.sha,
                "transcript_shape": (resp.usage or {}).get("transcript_shape"),
                "probe": None,
                "outcome": {"kind": "generation", "value": outcome.value,
                            "extra": outcome.extra},
                "text": resp.text, "usage": resp.usage,
                "timing_ms": resp.timing_ms, "error": resp.error,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if resp.error:
                n_err += 1
            if i % 50 == 0 or resp.error:
                print(f"[run {i}/{len(plan)}] {entry['surface']}/{entry['scheme']}/"
                      f"{entry['question']}/rep{entry['rep']} err={resp.error}", flush=True)
    print(f"[run] wrote {written} new records, {n_err} errors ({len(plan)} planned)",
          flush=True)
    return written


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def load_judges(path: str = JUDGES_PATH) -> Dict[str, Dict[str, Any]]:
    """trial_key -> judge row, last write winning (a re-judge corrects)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("trial_key"):
                    out[row["trial_key"]] = row
    return out


def judge_payload(surface: str, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Labels, their values on the rubric's own map, and one `political` score.

    `political` averages only the fields the rubric scores on the seven-point
    left/right map -- never the formality/optimism/concreteness controls, which
    exist to show that something did *not* move with the photos.
    """
    if not row:
        return None
    if row.get("error"):
        return {"error": row["error"]}
    labels = row.get("labels") or {}
    if not labels:
        return {"error": "judge row carried no labels"}
    spec = judge_specs().get("s2_proposal")
    values: Dict[str, Optional[float]] = {}
    political: List[float] = []
    for field in spec.fields:
        label = labels.get(field)
        mapping = spec.label_map.get(field)
        value = mapping.get(label) if (mapping and isinstance(label, str)) else None
        values[field] = value
        if mapping is LEAN_MAP and value is not None:
            political.append(value)
    return {
        "labels": {f: labels.get(f) for f in spec.fields},
        "values": values,
        "political": statistics.fmean(political) if political else None,
        "political_fields": [f for f in spec.fields if spec.label_map.get(f) is LEAN_MAP],
        "political_content_present": labels.get("political_content_present"),
        "refusal": labels.get("refusal"),
        "rationale": labels.get("rationale"),
        "cached": row.get("cached"),
    }


def phase_export(trials_path: str = TRIALS_PATH, out_dir: str = UPLOADS) -> Dict[str, Any]:
    rows = []
    with open(trials_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    judges = load_judges()

    records = []
    for r in rows:
        extra = (r.get("outcome") or {}).get("extra") or {}
        records.append({
            "id": r["trial_key"], "surface": r["surface"], "scheme": r["scheme"],
            "condition": r["condition"], "question": r["question"], "rep": r["rep"],
            "domain": r.get("domain"), "topic": r.get("topic"),
            "bucket": r["bucket"], "item_id": r["item_id"],
            "image_mean": r.get("image_mean"),
            "word_count": extra.get("word_count"),
            "rule_refusal": extra.get("refusal"),
            "truncated": (r.get("usage") or {}).get("finish_reason") == "length",
            "judge": judge_payload(r["surface"], judges.get(r["trial_key"])),
            "text": r.get("text"),
            "error": r.get("error"),
        })

    summary = _summarise(records)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "round15.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False)
    with open(os.path.join(out_dir, "round15_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[export] {len(records)} records, {len(summary['cells'])} cells -> {out_dir}")
    return summary


def _summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The cells the visualisation needs: (task, conversation, band) -> political."""
    cells = []
    by_cell: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["condition"] != "C":
            continue
        by_cell[(rec["surface"], rec["scheme"], rec["bucket"])].append(rec)
    for (surface, scheme, bucket), group in sorted(by_cell.items()):
        vals = [(g["judge"] or {}).get("political") for g in group]
        vals = [v for v in vals if v is not None]
        cells.append({
            "surface": surface, "scheme": scheme, "bucket": bucket,
            "n": len(group), "n_scored": len(vals),
            "political_mean": statistics.fmean(vals) if vals else None,
            "political_sd": statistics.pstdev(vals) if len(vals) > 1 else None,
        })

    baseline = []
    by_base: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["condition"] == "E":
            by_base[(rec["surface"], rec["scheme"])].append(rec)
    for (surface, scheme), group in sorted(by_base.items()):
        vals = [(g["judge"] or {}).get("political") for g in group]
        vals = [v for v in vals if v is not None]
        baseline.append({"surface": surface, "scheme": scheme, "n": len(group),
                         "political_mean": statistics.fmean(vals) if vals else None})

    # Repeatability: how often the same cell reproduced its text across reps.
    by_id: Dict[Any, List[str]] = defaultdict(list)
    for rec in records:
        by_id[(rec["surface"], rec["scheme"], rec["question"], rec["item_id"],
               rec["condition"])].append(rec["text"] or "")
    repeated = {k: v for k, v in by_id.items() if len(v) > 1}
    identical = sum(1 for v in repeated.values() if len(set(v)) == 1)

    return {
        "cells": cells,
        "baseline": baseline,
        "repeatability": {
            "cells_with_repeats": len(repeated),
            "cells_identical_across_reps": identical,
            "fraction_identical": (identical / len(repeated)) if repeated else None,
        },
        "counts": {
            "by_surface": dict(Counter(r["surface"] for r in records)),
            "by_condition": dict(Counter(r["condition"] for r in records)),
            "judged": sum(1 for r in records if (r["judge"] or {}).get("political") is not None),
            "judge_missing": sum(1 for r in records if r["judge"] is None),
            "judge_errors": sum(1 for r in records if (r["judge"] or {}).get("error")),
            "rule_refusals": sum(1 for r in records if r["rule_refusal"]),
            "truncated": sum(1 for r in records if r["truncated"]),
            "errors": sum(1 for r in records if r["error"]),
        },
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=["plan", "smoke", "run", "export"])
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the number of trials; 0 = no cap (and, for "
                             "--phase smoke, one per task/conversation/band/condition)")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="how many times each cell is measured")
    args = parser.parse_args()
    if args.phase == "plan":
        phase_plan(args.reps)
    elif args.phase == "smoke":
        phase_run(limit=args.limit, smoke=True, reps=args.reps)
    elif args.phase == "run":
        phase_run(limit=args.limit, reps=args.reps)
    elif args.phase == "export":
        phase_export()


if __name__ == "__main__":
    sys.exit(main())
