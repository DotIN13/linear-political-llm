"""Round 9 on the vLLM path: all six surfaces, no probe.

The HF pilot (``round8_pilot.py``) is the reference for the plan and the record
shape. Two things are deliberately different:

* **No calibrate phase.** The delivery-offset calibration is a probe reading, and
  this backend has no activations. The last measured value stands (MAD/sd =
  0.165, docs/bench/14) and belongs to the ``local_hf`` path.
* **One long job instead of a 28-minute chain.** Loading the server costs
  minutes; the run does not.

Phases: ``plan`` (prints the plan, touches nothing), ``run`` (needs a server),
``export`` (writes all6.json / all6_summary.json). ``plan`` and ``export`` are
CPU-only and are what the local test exercises.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from bench import registry
from bench.store import git_rev, measurement_rev, trial_key
from bench.surfaces.generation import SURFACE_IDS, build_scheme_messages
from bench.types import Conversation, Item, Trial

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "runs", "pilot_round9v")
TRIALS_PATH = os.path.join(OUT_DIR, "trials.jsonl")
ITEMS_FILE = os.path.join(ROOT, "items", "explore_bucket_v1.jsonl")
UPLOADS = "/project/jevans/tzhang3/agent-bridge-tmp/uploads/all6"

BUCKETS = ("low", "mid", "high")
ITEMS_PER_BUCKET = 6
BASELINE_SEEDS = (42, 43, 44, 45)
A_SEED = 42
BASELINE_ITEM_ID = "__baseline__"

# chat runs bare; agentic needs the prefill or it refuses outright (docs/bench/11:
# 9/9 refusals). Round 8 measured the cost of the prefill on chat: the three-bucket
# span fell from 0.943 to 0.53.
PREFILL_BY_SCHEME = {"chat": "off", "agentic": "on"}


def load_items() -> List[Dict[str, Any]]:
    per_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(ITEMS_FILE, encoding="utf-8") as handle:
        for line in handle:
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
def build_plan(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    registry.load_all()
    plan: List[Dict[str, Any]] = []

    for sid in SURFACE_IDS:
        surface = registry.get_surface(sid)()
        for scheme in ("chat", "agentic"):
            prefill = PREFILL_BY_SCHEME[scheme]
            for row in items:
                item = Item.from_dict(row)
                base = {"scheme": scheme, "prefill": prefill}
                fwd = surface.build(item, "C", dict(base), seed=A_SEED)
                plan.append(_entry(fwd, sid, "A", "C", scheme, prefill, row, item,
                                   order_arm="fwd" if fwd.variant.get("order") else None))
                # s3 only: the same item again with the order reversed, so the
                # position-1 primacy averages out instead of riding on the DV.
                order = fwd.variant.get("order")
                if order:
                    rev_variant = dict(base)
                    rev_variant["order"] = list(reversed(list(order)))
                    rev_variant["order_arm"] = "rev"
                    rev = surface.build(item, "C", rev_variant, seed=A_SEED)
                    plan.append(_entry(rev, sid, "A", "C", scheme, prefill, row, item,
                                       order_arm="rev"))

        # P arm: s1 on chat *with* the prefill, same items, to size the prefill's
        # compression cost within-item rather than across rounds.
        if sid == "s1_speech":
            for row in items:
                item = Item.from_dict(row)
                t = surface.build(item, "C", {"scheme": "chat", "prefill": "on"}, seed=A_SEED)
                plan.append(_entry(t, sid, "P", "C", "chat", "on", row, item))

        # C arm: no image, both schemes, four seeds.
        for scheme in ("chat", "agentic"):
            prefill = PREFILL_BY_SCHEME[scheme]
            question = surface.question(None, "shown", "v0")
            for seed in BASELINE_SEEDS:
                if scheme == "chat":
                    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
                    tools = None
                else:
                    messages, tools = build_scheme_messages("agentic", [], question)
                trial = Trial(
                    surface=sid, item_id=BASELINE_ITEM_ID, condition="E",
                    conversation=Conversation(messages=messages, images=[]),
                    candidates=[], probe_points=surface.probe_points(None),
                    max_new_tokens=surface.max_new_tokens,
                    variant={"scheme": scheme, "prefill": prefill},
                    meta={"family": "generation", "scheme": scheme, "prompt": "v0",
                          "prefill": surface.prefill_text if prefill == "on" else None,
                          "question": question, "tools": tools,
                          "prefix_n_messages": len(messages) - 1,
                          "condition_desc": "no-image baseline", "n_images": 0,
                          "item_invariant": True,
                          "judge": surface.judge_spec.id if surface.judge_spec else None},
                )
                plan.append({"trial": trial, "surface": sid, "arm": "C", "condition": "E",
                             "scheme": scheme, "prefill": prefill, "bucket": None,
                             "item_id": BASELINE_ITEM_ID, "image_mean": None,
                             "image_scores": [], "covariates": {}, "order_arm": None,
                             "seed": seed})
    return plan


def _smoke_slice(plan: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """One trial per (scheme, arm) combination, up to ``n`` -- agentic included."""
    picked: List[Dict[str, Any]] = []
    seen = set()
    for entry in plan:
        key = (entry["scheme"], entry["arm"], entry["order_arm"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(entry)
        if len(picked) >= n:
            break
    return picked


def _entry(trial: Trial, sid: str, arm: str, condition: str, scheme: str, prefill: str,
           row: Dict[str, Any], item: Item, order_arm: Optional[str] = None) -> Dict[str, Any]:
    return {"trial": trial, "surface": sid, "arm": arm, "condition": condition,
            "scheme": scheme, "prefill": prefill, "bucket": row["bucket"],
            "item_id": item.item_id, "image_mean": item.image_mean,
            "image_scores": list(item.image_scores), "covariates": dict(item.covariates),
            "order_arm": order_arm, "seed": A_SEED}


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def phase_run(adaptor: Any = None, limit: int = 0, smoke: bool = False) -> int:
    """``limit`` caps the number of trials; ``smoke`` picks a spread instead of a prefix.

    A prefix of the plan is all one surface and one scheme, which is the wrong
    thing to smoke-test: the risk is the *agentic* transcript, because folding
    tool turns into user turns is this backend's one hand-written step.
    """
    registry.load_all()
    items = load_items()
    plan = build_plan(items)
    if smoke:
        plan = _smoke_slice(plan, limit or 4)
    elif limit:
        plan = plan[:limit]

    if adaptor is None:
        from bench.adaptors.vllm_server import VLLMServerAdaptor
        adaptor = VLLMServerAdaptor(
            model=os.environ.get("VLLM_SERVED_MODEL", "qwen3-vl-8b-instruct"),
            base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8073/v1"),
            seed=A_SEED,
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
    written = 0
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
                "trial_key": key, "run_id": "pilot_round9v",
                "code_rev": code_rev, "measurement_rev": rev,
                "adaptor": adaptor.name, "model": adaptor.model,
                "surface": entry["surface"], "surface_family": "generation",
                "arm": entry["arm"], "condition": entry["condition"],
                "scheme": entry["scheme"], "prefill": entry["prefill"],
                "order_arm": entry["order_arm"], "bucket": entry["bucket"],
                "item_id": entry["item_id"], "seed": entry["seed"],
                "image_mean": entry["image_mean"], "image_scores": entry["image_scores"],
                "covariates": entry["covariates"], "variant": trial.variant,
                "conversation_sha": trial.conversation.sha,
                "transcript_shape": (resp.usage or {}).get("transcript_shape"),
                "probe": None,                     # no ACTIVATIONS on this backend
                "outcome": {"kind": "generation", "value": outcome.value,
                            "extra": outcome.extra},
                "text": resp.text, "usage": resp.usage,
                "timing_ms": resp.timing_ms, "error": resp.error,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if i % 25 == 0 or resp.error:
                print(f"[run {i}/{len(plan)}] {entry['surface']}/{entry['scheme']}/"
                      f"{entry['item_id']} err={resp.error}", flush=True)
    print(f"[run] wrote {written} new records ({len(plan)} planned)", flush=True)
    return written


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs2 = [p[0] for p in pairs]; ys2 = [p[1] for p in pairs]
    if len(set(ys2)) < 2 or len(set(xs2)) < 2:
        return None
    mx = statistics.fmean(xs2); my = statistics.fmean(ys2)
    num = sum((a - mx) * (b - my) for a, b in zip(xs2, ys2))
    den = (sum((a - mx) ** 2 for a in xs2) * sum((b - my) ** 2 for b in ys2)) ** 0.5
    return None if den == 0 else num / den


def phase_export(trials_path: str = TRIALS_PATH, out_dir: str = UPLOADS) -> Dict[str, Any]:
    rows = []
    with open(trials_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    records = []
    for r in rows:
        extra = (r.get("outcome") or {}).get("extra") or {}
        records.append({
            "id": r["trial_key"], "surface": r["surface"], "scheme": r["scheme"],
            "prefill": r["prefill"], "arm": r["arm"], "bucket": r["bucket"],
            "item_id": r["item_id"], "order_arm": r.get("order_arm"),
            "image_mean": r["image_mean"], "image_means": r["image_scores"],
            "categories": (r.get("covariates") or {}).get("categories", []),
            "n_objects": (r.get("covariates") or {}).get("n_objects"),
            "s_pre": None, "s_gen": None, "s_gen_first25": None, "s_gen_last25": None,
            "deterministic": {"primary": (r.get("outcome") or {}).get("value"), **extra},
            "judge": r.get("judge"),
            "refusal": extra.get("refusal"), "refusal_match": extra.get("refusal_match"),
            "word_count": extra.get("word_count"),
            "truncated": (r.get("usage") or {}).get("finish_reason") == "length",
            "transcript_shape": r.get("transcript_shape"),
            "text": r.get("text"),
        })

    cells = []
    by_cell: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["arm"] == "A":
            by_cell[(rec["surface"], rec["scheme"], rec["bucket"])].append(rec)
    for (surface, scheme, bucket), group in sorted(by_cell.items()):
        prim = [g["deterministic"].get("primary") for g in group]
        prim = [p for p in prim if isinstance(p, (int, float))]
        cells.append({"surface": surface, "scheme": scheme, "bucket": bucket,
                      "n": len(group),
                      "primary_mean": statistics.fmean(prim) if prim else None,
                      "refusal_rate": statistics.fmean([1.0 if g["refusal"] else 0.0
                                                        for g in group]),
                      "s_gen_mean": None, "s_pre_mean": None})

    pearson = []
    by_sc: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec["arm"] == "A":
            by_sc[(rec["surface"], rec["scheme"])].append(rec)
    for (surface, scheme), group in sorted(by_sc.items()):
        pearson.append({
            "surface": surface, "scheme": scheme, "n": len(group),
            "image_mean_vs_primary": _pearson(
                [g["image_mean"] for g in group],
                [g["deterministic"].get("primary") for g in group]),
            "image_mean_vs_s_gen": None,
        })

    s3 = [r for r in records if r["surface"] == "s3_digest" and r["arm"] == "A"]
    position_rates: Dict[str, Any] = {}
    for arm in ("fwd", "rev"):
        picks = [r["deterministic"].get("picked_positions") or []
                 for r in s3 if r["order_arm"] == arm]
        picks = [p for p in picks if p]
        if picks:
            position_rates[arm] = [
                sum(1 for p in picks if pos in p) / len(picks) for pos in range(1, 13)]
    if "fwd" in position_rates and "rev" in position_rates:
        position_rates["balanced"] = [
            (a + b) / 2 for a, b in zip(position_rates["fwd"], position_rates["rev"])]

    summary = {"cells": cells, "pearson": pearson,
               "baseline": [{"surface": r["surface"], "scheme": r["scheme"],
                             "primary": r["deterministic"].get("primary"),
                             "refusal": r["refusal"]}
                            for r in records if r["arm"] == "C"],
               "s3_position_rates": position_rates,
               "counts": dict(Counter((r["surface"], r["scheme"], r["arm"]) for r in records
                                      ).most_common()) and
                         {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in
                          Counter((r["surface"], r["scheme"], r["arm"]) for r in records).items()},
               "note": ("adaptor=vllm: no ACTIVATIONS, so s_pre/s_gen/s_img are null "
                        "by construction. image_mean is precomputed, so the IV is intact.")}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all6.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False)
    with open(os.path.join(out_dir, "all6_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[export] {len(records)} records, {len(cells)} cells -> {out_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="plan", choices=("plan", "run", "export", "smoke"))
    parser.add_argument("--limit", type=int, default=0, help="cap the number of trials")
    args = parser.parse_args()
    if args.phase == "plan":
        plan = build_plan(load_items())
        by = Counter((p["surface"], p["scheme"], p["arm"]) for p in plan)
        for key in sorted(by):
            print(f"  {key[0]:14} {key[1]:8} {key[2]}  {by[key]:3d}")
        print(f"total {len(plan)}")
    elif args.phase == "run":
        phase_run(limit=args.limit)
    elif args.phase == "smoke":
        phase_run(limit=args.limit or 4, smoke=True)
    else:
        phase_export()


if __name__ == "__main__":
    main()
