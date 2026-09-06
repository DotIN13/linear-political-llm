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
from bench.judges.specs import LEAN_MAP, judge_specs
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

# chat runs bare; agentic asked for the prefill because docs/bench/11 measured 9/9
# refusals without it. But **only s1_speech was ever given a `prefill_text`** --
# the other five surfaces default to None, so `prefill="on"` there produced
# `meta["prefill"] = None` and a silently inert trial (and, since the guard
# landed, a crash instead: run 57894892 died on s2_proposal's first agentic
# trial, 80 records in).
#
# The honest fix is not to invent five prefill strings. It is to notice that the
# prefill is an expensive intervention -- round 8: it cut s1's three-bucket span
# from 0.943 to 0.53, so 44% of the effect -- justified by a refusal problem that
# has only ever been measured on s1, and that the round-9 smoke's single
# no-prefill agentic trial did *not* reproduce. So: agentic runs with the prefill
# where a prefill exists, bare where it does not, and the refusal rate is the
# finding rather than the thing we paper over.
PREFILL_BY_SCHEME = {"chat": "off", "agentic": "on"}


def prefill_of(surface: Any) -> str:
    """Whether this surface prefills, as a label for the records.

    Prefill stopped being a handle: a surface either declares ``prefill_text``
    (applied on every trial, both schemes) or it does not. This is reporting, not
    a choice -- there is nothing left to pass into ``build``.
    """
    return "on" if getattr(surface, "prefill_text", None) else "off"


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
            prefill = prefill_of(surface)
            for row in items:
                item = Item.from_dict(row)
                base = {"scheme": scheme}
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

        # The P and R arms are gone. Both were prefill on/off contrasts on s1 -- P
        # was chat *with* the prefill, R was agentic *without* it -- and prefill is
        # no longer a handle, so neither is expressible: s1 has a prefill_text and
        # therefore always prefills, on both schemes.
        #
        # R had already done its job. It was built to test docs/bench/11's claim of
        # 9/9 agentic refusals without the prefill and returned 0/18, which is why
        # the handle went away. Keeping a one-sided arm would just re-measure the A
        # arm under a second name.

        # C arm: no image, both schemes, four seeds.
        for scheme in ("chat", "agentic"):
            prefill = prefill_of(surface)
            question = surface.question(None, "shown", surface.question_ids()[0])
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
    """One trial per (surface, scheme, arm, order_arm), up to ``n``.

    **Surface is in the key on purpose.** This keyed on (scheme, arm, order_arm)
    alone, and since ``build_plan`` walks surface-by-surface every one of those
    keys is first satisfied by ``s1_speech`` -- so a 4-trial smoke was four
    s1 trials, and a per-surface configuration problem was invisible to it. That
    is exactly how the missing ``prefill_text`` on the other five surfaces got
    past a green smoke and killed the full run 80 records in.

    A full sweep is 26 keys; ``n=0`` means all of them, and any smaller ``n``
    takes them in plan order, which is still surface-major -- so pass 0, or at
    least enough to reach the last surface.
    """
    picked: List[Dict[str, Any]] = []
    seen = set()
    for entry in plan:
        key = (entry["surface"], entry["scheme"], entry["arm"], entry["order_arm"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(entry)
        if n and len(picked) >= n:
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
        plan = _smoke_slice(plan, limit)      # limit 0 = one per (surface, scheme, arm)
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
            # A prefill-on trial whose prefill never reached the server is not a
            # warning, it is the run being inert: the P arm *is* the prefill, and
            # so is the agentic refusal mitigation. Stop on the first one rather
            # than write 318 records that look fine. `is False` on purpose -- an
            # adaptor that does not report the flag is not being accused.
            if entry["prefill"] == "on" and not resp.error \
                    and (resp.usage or {}).get("prefill_applied") is False:
                raise RuntimeError(
                    f"prefill=on but the backend did not apply it "
                    f"({entry['surface']}/{entry['scheme']}); stopping before "
                    f"writing inert trials")
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


JUDGES_PATH = os.path.join(OUT_DIR, "judges.jsonl")


def load_judges(path: str = JUDGES_PATH) -> Dict[str, Dict[str, Any]]:
    """``trial_key -> judge labels``, last write winning.

    ``judges.jsonl`` is appended to, so a re-judge leaves both rows; taking the
    last one means a re-run corrects rather than duplicates. Rows carrying an
    ``error`` are kept -- a judge that failed is not the same as one that was
    never asked, and the export says which.
    """
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
    """One record's judge block: every label, its value on the spec's own map,
    and a single ``political`` score.

    ``political`` is the mean of the fields the spec scores on the seven-point
    left/right map -- so s1's five political axes, s2/s5's three, s4's one --
    and it is deliberately *not* the style controls (formality / optimism /
    concreteness), which are there to show that something did *not* move.
    Averaging gives every surface one number on the same -1..+1 scale, which is
    the only way six tasks go on one chart.
    """
    if not row:
        return None
    if row.get("error"):
        return {"error": row["error"]}
    labels = row.get("labels") or row.get("parsed") or {}
    if not labels:
        return {"error": "judge row carried no labels"}
    spec = judge_specs().get(surface)
    if spec is None:
        return {"labels": labels}
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
            "prefill": r["prefill"], "arm": r["arm"], "bucket": r["bucket"],
            "item_id": r["item_id"], "order_arm": r.get("order_arm"),
            "image_mean": r["image_mean"], "image_means": r["image_scores"],
            "categories": (r.get("covariates") or {}).get("categories", []),
            "n_objects": (r.get("covariates") or {}).get("n_objects"),
            "s_pre": None, "s_gen": None, "s_gen_first25": None, "s_gen_last25": None,
            "deterministic": {"primary": (r.get("outcome") or {}).get("value"), **extra},
            "judge": judge_payload(r["surface"], judges.get(r["trial_key"])),
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
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the number of trials; 0 = no cap (and, for "
                             "--phase smoke, one trial per surface/scheme/arm)")
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
        # `or 4` here defeated `--limit 0`, which _smoke_slice reads as "one per
        # (surface, scheme, arm)" -- so `smoke 0` quietly ran 4 s1-only trials, the
        # exact prefix behaviour 7d90b52 was meant to remove. The default lives in
        # the flag, not here.
        phase_run(limit=args.limit, smoke=True)
    else:
        phase_export()


if __name__ == "__main__":
    main()
