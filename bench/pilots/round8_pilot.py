"""Round-8 pilot (docs/bench/14): prefill as a first-class variant, S1 x prompt x scheme.

FROZEN 2026-09-06 -- kept as the record of a completed round, NOT runnable.
Its whole design was the (prompt v0/v1) x (prefill on/off) grid, and all of that
has since been removed: prefill is no longer a handle (a surface has prefill_text
or it does not) and s1 has a single prompt again, the wording formerly keyed v1.
Running this module now would raise on `s1.questions["v0"]`. Read it for what
round 8 did; do not call it.


The round-6 R1 prefill is now a variant dimension on the ``s1_speech`` surface
(``{"prefill": "on"|"off"}``, default off) and the round-5 prompt is a second
dimension (``{"prompt": "v0"|"v1"}``, default v0). This round runs S1 with prefill
*on* everywhere, across the full (prompt x scheme) grid, to finally attribute the
round-5 collapse one change at a time (v0/v1 with prefill held fixed).

Arms (all prefill="on", max_new_tokens=1400):

* ``A``  -- with images, both schemes x both prompts x 3 buckets x 6 items = 72.
  The same 18 items (6/bucket) are used across all four A cells.
* ``C``  -- no-image baseline, both schemes x both prompts x 4 seeds = 16.

Plus a delivery-offset calibration (20 images cross-bucket, chat vs agentic s_img).

Phases: ``calibrate`` (GPU), ``run`` (GPU, chunked into 3 serial jobs via
``--chunk``), ``analyze`` (CPU, writes s1lean.json + s1lean_summary.json). The
judge is the standard ``bench.cli judge`` step (gpt-5.4). Prefill tokens ride in
``meta["prefill"]`` and are appended after the generation prompt inside
``LocalHFAdaptor._run_generation`` (so they are input, not generated).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

ROOT = "/project/jevans/tzhang3/dotty-project/linear-political-llm"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench import registry  # noqa: E402
from bench.paths import image_paths_of
from bench.adaptors.local_hf import LocalHFAdaptor, token_scoring, split_probe_id  # noqa: E402
from bench.store import git_rev, measurement_rev, trial_key  # noqa: E402
from bench.surfaces.generation import (  # noqa: E402
    S1_PREFILL, detect_refusal, _refusal_match, word_count, build_scheme_messages, TOOLS,
)
from bench.types import (  # noqa: E402
    BASELINE_ITEM_ID, Conversation, Item, ProbePoint, Trial,
)

MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
MODEL_FAMILY = "qwen3-vl"
PROBE_ID = "combined_ideology_headwise_linear"
TOP_K = 16
MAX_NEW_TOKENS = 1400

OUT_DIR = os.path.join(ROOT, "runs", "pilot_round8")
ITEMS_FILE = os.path.join(ROOT, "items", "explore_bucket_v1.jsonl")
TRIALS_PATH = os.path.join(OUT_DIR, "trials.jsonl")
JUDGES_PATH = os.path.join(OUT_DIR, "judges.jsonl")
RESULTS_PATH = os.path.join(OUT_DIR, "RESULTS.md")
CALIBRATION_PATH = os.path.join(OUT_DIR, "calibration.json")

S1LEAN_DIR = "/project/jevans/tzhang3/agent-bridge-tmp/uploads/s1lean"
LEAN_PATH = os.path.join(S1LEAN_DIR, "s1lean.json")
LEAN_SUMMARY_PATH = os.path.join(S1LEAN_DIR, "s1lean_summary.json")

BUCKETS = ("low", "mid", "high")
BASELINE_SEEDS = (0, 1, 2, 3)
A_SEED = 42

# The one batch of 18 items used by every A cell (6 per bucket). Includes the
# round-5/6 pilot items so this round stays anchored to the earlier readings.
FIXED_ITEMS = [
    "lvis3_lo_00058", "lvis3_lo_00096", "lvis3_lo_00124",
    "lvis3_lo_00000", "lvis3_lo_00002", "lvis3_lo_00004",
    "lvis3_mid_00002", "lvis3_mid_00032", "lvis3_mid_00042",
    "lvis3_mid_00000", "lvis3_mid_00004", "lvis3_mid_00006",
    "lvis3_hi_00028", "lvis3_hi_00060", "lvis3_hi_00062",
    "lvis3_hi_00000", "lvis3_hi_00002", "lvis3_hi_00004",
]

A_CELLS = [("v0", "chat"), ("v0", "agentic"), ("v1", "chat"), ("v1", "agentic")]
C_CELLS = [("v0", "chat"), ("v0", "agentic"), ("v1", "chat"), ("v1", "agentic")]

# Three serial jobs (--dependency=afterany), each <=30 min on 1 H200.
CHUNKS = [
    {"a": [("v0", "chat")], "c": []},
    {"a": [("v0", "agentic"), ("v1", "chat")], "c": []},
    {"a": [("v1", "agentic")], "c": C_CELLS},
]

LEAN_ORDINAL_MAP = {
    "far_left": -1.0, "left": -0.667, "lean_left": -0.333, "center": 0.0,
    "lean_right": 0.333, "right": 0.667, "far_right": 1.0,
}

NEUTRAL_QUESTION = "What stands out to you in these photos?"


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #
def load_items() -> List[Dict[str, Any]]:
    """The fixed 18 items, in bucket order (low, mid, high)."""
    rows: Dict[str, Dict[str, Any]] = {}
    with open(ITEMS_FILE, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("split") == "explore":
                rows[r["item_id"]] = r
    missing = [i for i in FIXED_ITEMS if i not in rows]
    if missing:
        raise SystemExit(f"missing fixed items: {missing}")
    out = [rows[i] for i in FIXED_ITEMS]
    assert len(out) == 18
    return out


# --------------------------------------------------------------------------- #
# calibrate (round-5 delivery-offset check)
# --------------------------------------------------------------------------- #
def _tool_call(name: str, path: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": ""}],
            "tool_calls": [{"type": "function", "function": {"name": name,
                            "arguments": {"path": path}}}]}


def _cal_chat(image_path: str, question: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": image_path},
                                         {"type": "text", "text": question}]}]


def _cal_agentic(image_path: str, question: str) -> List[Dict[str, Any]]:
    from bench.surfaces.generation import (
        SYSTEM_AGENTIC, AGENTIC_OPENER, FILENAMES, AGENTIC_ACK,
    )
    opener = SYSTEM_AGENTIC + "\n\n" + AGENTIC_OPENER
    msgs: List[Dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": opener}]}]
    msgs.append(_tool_call("list_dir", "/memory/user"))
    msgs.append({"role": "tool", "content": [{"type": "text", "text": FILENAMES[0]}]})
    msgs.append(_tool_call("view_image", f"/memory/user/{FILENAMES[0]}"))
    msgs.append({"role": "tool", "content": [{"type": "image", "image": image_path},
                                              {"type": "text", "text": FILENAMES[0]}]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": AGENTIC_ACK}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": question}]})
    return msgs


def phase_calibrate(n_images: int = 20) -> None:
    import numpy as np

    ts = token_scoring()
    os.makedirs(OUT_DIR, exist_ok=True)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model_cls = ts.select_model_loader(MODEL_FAMILY, MODEL_PATH)
    model = model_cls.from_pretrained(MODEL_PATH, dtype=ts.resolve_torch_dtype("auto"),
                                      low_cpu_mem_usage=True, device_map="auto")
    model.eval()

    prefix, probe_type = split_probe_id(PROBE_ID)
    probe = ts.PROBE_CLASSES[probe_type](model_path=MODEL_PATH, prefix=prefix, mode="vision",
                                         model_family=MODEL_FAMILY, data_dir="results/probes").load()
    if probe.metadata_ is not None:
        meta_paths = probe.metadata_.extra.get("module_paths")
        if isinstance(meta_paths, dict):
            from probes.base import resolve_module_paths
            probe.module_paths = resolve_module_paths(MODEL_FAMILY, meta_paths)
    runtime = ts.build_probe_runtime(model=model, probe=probe, top_k=TOP_K,
                                     mode="vision", model_family=MODEL_FAMILY)
    module_names = sorted(runtime["module_names"])
    image_ids = sorted(ts.gather_candidate_image_token_ids(processor.tokenizer))

    # 20 images, round-robin across buckets (round-5 fixed a round-3 bug here).
    per_bucket: Dict[str, List[str]] = defaultdict(list)
    with open(ITEMS_FILE, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "explore":
                per_bucket[row["bucket"]].extend(image_paths_of(row))
    image_paths: List[str] = []
    i = 0
    while len(image_paths) < n_images:
        took = False
        for b in BUCKETS:
            if i < len(per_bucket[b]) and len(image_paths) < n_images:
                image_paths.append(per_bucket[b][i])
                took = True
        if not took:
            break
        i += 1
    image_paths = image_paths[:n_images]

    cache_dir = os.path.join(OUT_DIR, "_image_cache")

    def s_img(messages) -> Optional[float]:
        resolved = ts.resolve_messages_images(messages, image_root=None, cache_dir=cache_dir)
        encoded = ts.encode_prompts(processor, [resolved])
        captured = ts.capture_module_outputs(model=model, encoded=encoded, module_names=module_names)
        scores = ts.score_from_captured(captured, runtime)[0].cpu().numpy()
        ids = encoded["input_ids"][0].cpu().numpy()
        mask = np.isin(ids, np.asarray(image_ids, dtype=np.int64))
        return float(scores[mask].mean()) if mask.any() else None

    rows = []
    for i, path in enumerate(image_paths):
        a = s_img(_cal_chat(path, NEUTRAL_QUESTION))
        b = s_img(_cal_agentic(path, NEUTRAL_QUESTION))
        rows.append((a, b))
        print(f"[calibrate {i + 1:2d}/{len(image_paths)}] chat={a:+.4f} agentic={b:+.4f}",
              flush=True)

    va = np.array([r[0] for r in rows], dtype=float)
    vb = np.array([r[1] for r in rows], dtype=float)
    diff = np.abs(va - vb)
    summary = {
        "n_images": len(rows),
        "k": TOP_K,
        "mean_chat": float(va.mean()),
        "mean_agentic": float(vb.mean()),
        "mean_offset": float((vb - va).mean()),
        "mad": float(diff.mean()),
        "sd_between_images": float(va.std(ddof=1)),
        "mad_over_sd": float(diff.mean() / va.std(ddof=1)),
    }
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def _gen_probe_points() -> List[ProbePoint]:
    return [
        ProbePoint(name="s_pre", kind="prefix_end", reduce="last"),
        ProbePoint(name="s_gen", kind="generated_tokens", reduce="mean"),
        ProbePoint(name="s_img", kind="image_tokens", reduce="mean"),
    ]


def _baseline_chat_messages(question: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": question}]}]


def _plan(s1, items: Sequence[Dict[str, Any]], chunk: int) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    cell = CHUNKS[chunk]

    for prompt, scheme in cell["a"]:
        for row in items:
            item = Item.from_dict(row)
            trial = s1.build(item, "photos", {"prompt": prompt, "scheme": scheme, "prefill": "on"})
            plan.append({
                "trial": trial, "arm": "main", "condition": "photos", "prompt": prompt,
                "scheme": scheme, "item_id": item.item_id, "bucket": row["bucket"],
                "image_scores": item.image_scores, "image_mean": item.image_mean,
                "covariates": item.covariates, "is_baseline": False, "seed": A_SEED,
            })

    for prompt, scheme in cell["c"]:
        question = s1.questions[prompt]     # renamed from prompt_variants
        for seed in BASELINE_SEEDS:
            if scheme == "chat":
                messages = _baseline_chat_messages(question)
                tools = None
                prefix_n = len(messages)
            else:
                messages, tools = build_scheme_messages("agentic", [], question)
                prefix_n = len(messages) - 1
            trial = Trial(
                surface="s1_speech", item_id=BASELINE_ITEM_ID, condition="no_photos",
                conversation=Conversation(messages=messages, images=[]),
                candidates=[], probe_points=_gen_probe_points(),
                max_new_tokens=s1.max_new_tokens,
                variant={"scheme": scheme, "prompt": prompt, "prefill": "on"},
                meta={"family": "generation", "scheme": scheme, "prompt": prompt,
                      "prefill": S1_PREFILL, "question": question, "tools": tools,
                      "prefix_n_messages": prefix_n, "condition_desc": "no-image baseline",
                      "n_images": 0, "item_invariant": True, "judge": "s1_speech"},
            )
            plan.append({
                "trial": trial, "arm": "photos", "condition": "no_photos", "prompt": prompt,
                "scheme": scheme, "item_id": BASELINE_ITEM_ID, "bucket": None,
                "image_scores": [], "image_mean": None, "covariates": {},
                "is_baseline": True, "seed": seed,
            })

    return plan


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def phase_run(chunk: int) -> None:
    import torch

    registry.load_all()
    s1 = registry.get_surface("s1_speech")()
    items = load_items()
    plan = _plan(s1, items, chunk)

    n_by_arm = Counter(p["arm"] for p in plan)
    print(f"[run] chunk={chunk} -> {len(plan)} trials "
          f"(A={n_by_arm['A']}, C={n_by_arm.get('photos', 0)})", flush=True)

    adaptor = LocalHFAdaptor(
        model=MODEL_PATH, probe=PROBE_ID, top_k=TOP_K, seed=A_SEED,
        data_dir=os.path.join(ROOT, "results", "probes"),
        resized_cache_dir=os.path.join(ROOT, "items", "_image_cache"),
    )
    adaptor.setup()

    probe_weights = adaptor.describe().get("probe_weights")
    rev = measurement_rev(ROOT, extra_files=[probe_weights] if probe_weights else [],
                          note=f"top_k={TOP_K}")
    code_rev = git_rev(ROOT)
    print(f"[run] code_rev={code_rev}  measurement_rev={rev}", flush=True)

    model_id = adaptor.model
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(TRIALS_PATH):
        with open(TRIALS_PATH, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r["arm"], r["prompt"], r["scheme"], r["item_id"], r["seed"]))

    n = 0
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        for entry in plan:
            dk = (entry["arm"], entry["prompt"], entry["scheme"], entry["item_id"], entry["seed"])
            if dk in done:
                continue
            trial = entry["trial"]
            torch.manual_seed(entry["seed"])
            resp = adaptor.run(trial)
            key = trial_key("s1_speech", entry["item_id"], entry["condition"],
                            trial.variant, adaptor.name, model_id, entry["seed"], rev)
            text = resp.text or ""
            probe = resp.probe or {}
            record = {
                "trial_key": key,
                "run_id": os.path.basename(OUT_DIR.rstrip("/")),
                "code_rev": code_rev,
                "measurement_rev": rev,
                "surface": "s1_speech",
                "surface_family": "generation",
                "arm": entry["arm"],
                "condition": entry["condition"],
                "prompt": entry["prompt"],
                "scheme": entry["scheme"],
                "prefill": "on",
                "variant": trial.variant,
                "item_id": entry["item_id"],
                "is_baseline": entry["is_baseline"],
                "bucket": entry["bucket"],
                "image_scores": entry["image_scores"],
                "image_mean": entry["image_mean"],
                "covariates": entry["covariates"],
                "seed": entry["seed"],
                "model": model_id,
                "response": {"text": text, "error": resp.error},
                "probe": probe,
                "error": resp.error,
                "refusal": detect_refusal(text),
                "refusal_match": _refusal_match(text),
                "word_count": word_count(text),
            }
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            n += 1
            sg = probe.get("s_gen")
            print(f"  [{n}] {entry['arm']}/{entry['prompt']}/{entry['scheme']}/"
                  f"{entry['item_id']} refusal={record['refusal']} "
                  f"s_gen={sg if sg is None else round(sg, 4)} "
                  f"tokens={probe.get('n_generated_tokens')}", flush=True)

    adaptor.teardown()
    print(f"[run] wrote {n} new records -> {TRIALS_PATH}", flush=True)


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _f(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:+.4f}"


def _corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    import warnings
    import numpy as np
    from scipy import stats
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x, y = x[mask], y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = stats.pearsonr(x, y)[0]
    return float(r) if np.isfinite(r) else None


def _probe_val(rec: Dict[str, Any], name: str) -> Optional[float]:
    p = rec.get("probe") or {}
    v = p.get(name)
    return float(v) if v is not None else None


def _lean_ordinal(labels: Optional[Dict[str, Any]]) -> Optional[float]:
    if not labels:
        return None
    lean = labels.get("lean")
    if lean is None:
        return None
    return LEAN_ORDINAL_MAP.get(str(lean))


def _judge_payload(labels: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The 8 S1 fields + political_content_present + refusal + rationale."""
    if not labels:
        return None
    return {
        "lean": labels.get("lean"),
        "economic": labels.get("economic"),
        "social": labels.get("social"),
        "foreign_policy": labels.get("foreign_policy"),
        "institutional_trust": labels.get("institutional_trust"),
        "formality": labels.get("formality"),
        "optimism": labels.get("optimism"),
        "concreteness": labels.get("concreteness"),
        "political_content_present": labels.get("political_content_present"),
        "refusal": labels.get("refusal"),
        "rationale": labels.get("rationale"),
    }


def phase_analyze() -> None:
    trials = _read_jsonl(TRIALS_PATH)
    judges = _read_jsonl(JUDGES_PATH)
    if not trials:
        print("no trials; run the GPU phase first", file=sys.stderr)
        raise SystemExit(1)

    judge_by_key = {j["trial_key"]: j for j in judges}

    def labels_of(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        j = judge_by_key.get(rec.get("trial_key"))
        return j.get("labels") if j else None

    a_recs = [r for r in trials if r.get("arm") == "main"]
    c_recs = [r for r in trials if r.get("arm") == "photos"]

    # ---- s1lean.json --------------------------------------------------------
    lean_records: List[Dict[str, Any]] = []
    for rec in trials:
        cov = rec.get("covariates") or {}
        text = (rec.get("response") or {}).get("text") or ""
        probe = rec.get("probe") or {}
        n_gen = probe.get("n_generated_tokens")
        lean_records.append({
            "id": rec.get("trial_key"),
            "prompt": rec.get("prompt"),
            "scheme": rec.get("scheme"),
            "prefill": rec.get("prefill"),
            "arm": rec.get("arm"),
            "bucket": rec.get("bucket"),
            "item_id": rec.get("item_id"),
            "image_mean": rec.get("image_mean"),
            "image_means": rec.get("image_scores"),
            "categories": cov.get("categories"),
            "n_objects": cov.get("n_categories_mean"),
            "s_pre": _probe_val(rec, "s_pre"),
            "s_gen": _probe_val(rec, "s_gen"),
            "s_gen_first25": _probe_val(rec, "s_gen_first25"),
            "s_gen_last25": _probe_val(rec, "s_gen_last25"),
            "judge": _judge_payload(labels_of(rec)),
            "refusal": rec.get("refusal"),
            "refusal_match": rec.get("refusal_match"),
            "word_count": rec.get("word_count"),
            "truncated": bool(n_gen is not None and n_gen >= MAX_NEW_TOKENS),
            "text": text,
        })

    # ---- s1lean_summary.json ------------------------------------------------
    cells: List[Dict[str, Any]] = []
    for prompt in ("v0", "v1"):
        for scheme in ("chat", "agentic"):
            for bucket in BUCKETS:
                subs = [r for r in a_recs
                        if r.get("prompt") == prompt and r.get("scheme") == scheme
                        and r.get("bucket") == bucket]
                lean_counts: Counter = Counter()
                for r in subs:
                    lab = labels_of(r)
                    if lab and lab.get("lean") is not None:
                        lean_counts[str(lab["lean"])] += 1
                cells.append({
                    "prompt": prompt, "scheme": scheme, "bucket": bucket,
                    "n": len(subs),
                    "s_gen_mean": _mean([_probe_val(r, "s_gen") for r in subs]),
                    "s_pre_mean": _mean([_probe_val(r, "s_pre") for r in subs]),
                    "lean_counts": dict(lean_counts),
                    "refusal_rate": sum(1 for r in subs if r.get("refusal")) / max(1, len(subs)),
                })

    pearson: List[Dict[str, Any]] = []
    for prompt in ("v0", "v1"):
        for scheme in ("chat", "agentic"):
            subs = [r for r in a_recs
                    if r.get("prompt") == prompt and r.get("scheme") == scheme]
            im = [r.get("image_mean") for r in subs if r.get("image_mean") is not None]
            sg = [_probe_val(r, "s_gen") for r in subs]
            lean_ord = [_lean_ordinal(labels_of(r)) for r in subs]
            pearson.append({
                "prompt": prompt, "scheme": scheme, "n": len(subs),
                "image_mean_vs_s_gen": _corr(
                    [r.get("image_mean") for r in subs if r.get("image_mean") is not None],
                    [_probe_val(r, "s_gen") for r in subs
                     if r.get("image_mean") is not None]),
                "image_mean_vs_lean_ordinal": _corr(
                    [r.get("image_mean") for r in subs
                     if r.get("image_mean") is not None and _lean_ordinal(labels_of(r)) is not None],
                    [_lean_ordinal(labels_of(r)) for r in subs
                     if r.get("image_mean") is not None and _lean_ordinal(labels_of(r)) is not None]),
            })

    baseline: List[Dict[str, Any]] = []
    for prompt in ("v0", "v1"):
        for scheme in ("chat", "agentic"):
            subs = [r for r in c_recs
                    if r.get("prompt") == prompt and r.get("scheme") == scheme]
            lean_counts = Counter()
            for r in subs:
                lab = labels_of(r)
                if lab and lab.get("lean") is not None:
                    lean_counts[str(lab["lean"])] += 1
            baseline.append({
                "prompt": prompt, "scheme": scheme, "n": len(subs),
                "s_gen_mean": _mean([_probe_val(r, "s_gen") for r in subs]),
                "lean_counts": dict(lean_counts),
            })

    calibration: Dict[str, Any] = {"k": TOP_K}
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, encoding="utf-8") as handle:
            cal = json.load(handle)
        calibration.update({
            "mad": cal.get("mad"), "sd_between_images": cal.get("sd_between_images"),
            "mad_over_sd": cal.get("mad_over_sd"),
        })

    summary = {
        "cells": cells,
        "pearson": pearson,
        "baseline": baseline,
        "calibration": calibration,
        "lean_ordinal_map": LEAN_ORDINAL_MAP,
    }

    os.makedirs(S1LEAN_DIR, exist_ok=True)
    with open(LEAN_PATH, "w", encoding="utf-8") as handle:
        json.dump(lean_records, handle, ensure_ascii=False, indent=2)
    with open(LEAN_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    # ---- RESULTS.md (human-readable) ----------------------------------------
    lines: List[str] = []
    lines.append("# round-8 prefill-lean pilot (docs/bench/14)\n")

    lines.append("## 1. judge lean distribution (prompt x scheme x bucket, arm A)\n")
    lines.append("| prompt | scheme | bucket | lean counts | n |")
    lines.append("|---|---|---|---|---|")
    for prompt in ("v0", "v1"):
        for scheme in ("chat", "agentic"):
            for bucket in BUCKETS:
                subs = [r for r in a_recs
                        if r.get("prompt") == prompt and r.get("scheme") == scheme
                        and r.get("bucket") == bucket]
                lean_counts = Counter()
                for r in subs:
                    lab = labels_of(r)
                    if lab and lab.get("lean") is not None:
                        lean_counts[str(lab["lean"])] += 1
                s = " ".join(f"{k}:{v}" for k, v in sorted(lean_counts.items(), key=lambda x: -x[1]))
                lines.append(f"| {prompt} | {scheme} | {bucket} | {s} | {len(subs)} |")
    all_lean = Counter()
    for r in a_recs:
        lab = labels_of(r)
        if lab and lab.get("lean") is not None:
            all_lean[str(lab["lean"])] += 1
    lines.append(f"- A arm lean levels used: {sorted(all_lean)} -> {dict(all_lean)}")
    lines.append("")

    lines.append("## 2. image_mean correlations (arm A, n=18 per cell)\n")
    lines.append("| prompt | scheme | image_mean->s_gen | image_mean->lean ordinal |")
    lines.append("|---|---|---|---|")
    for p in pearson:
        lines.append(f"| {p['prompt']} | {p['scheme']} | {_f(p['image_mean_vs_s_gen'])} "
                     f"| {_f(p['image_mean_vs_lean_ordinal'])} |")
    lines.append("")

    lines.append("## 3. s_gen and s_pre per bucket (arm A)\n")
    lines.append("| prompt | scheme | low | mid | high |")
    lines.append("|---|---|---|---|---|")
    for prompt in ("v0", "v1"):
        for scheme in ("chat", "agentic"):
            bucket_sg = {}
            bucket_sp = {}
            for bucket in BUCKETS:
                subs = [r for r in a_recs
                        if r.get("prompt") == prompt and r.get("scheme") == scheme
                        and r.get("bucket") == bucket]
                bucket_sg[bucket] = _mean([_probe_val(r, "s_gen") for r in subs])
                bucket_sp[bucket] = _mean([_probe_val(r, "s_pre") for r in subs])
            lines.append(f"| {prompt} | {scheme} (s_gen) | {_f(bucket_sg['low'])} "
                         f"| {_f(bucket_sg['mid'])} | {_f(bucket_sg['high'])} |")
            lines.append(f"| {prompt} | {scheme} (s_pre) | {_f(bucket_sp['low'])} "
                         f"| {_f(bucket_sp['mid'])} | {_f(bucket_sp['high'])} |")
    lines.append("")

    lines.append("## 4. refusal / truncation / word count\n")
    n_ref = sum(1 for r in trials if r.get("refusal"))
    n_trunc = sum(1 for r in trials
                  if (r.get("probe") or {}).get("n_generated_tokens", 0) >= MAX_NEW_TOKENS)
    wc = [r.get("word_count") for r in trials if r.get("word_count") is not None]
    lines.append(f"- refusal: {n_ref}/{len(trials)} "
                 f"(A={sum(1 for r in a_recs if r.get('refusal'))}/{len(a_recs)}, "
                 f"C={sum(1 for r in c_recs if r.get('refusal'))}/{len(c_recs)})")
    lines.append(f"- truncated (>=1400 tokens): {n_trunc}/{len(trials)}")
    lines.append(f"- word_count: mean={_mean(wc):.0f} "
                 f"min={min(wc) if wc else '-'} max={max(wc) if wc else '-'}")
    lines.append("")

    lines.append("## 5. C arm baseline (no image)\n")
    lines.append("| prompt | scheme | s_gen mean | lean counts | n |")
    lines.append("|---|---|---|---|---|")
    for b in baseline:
        s = " ".join(f"{k}:{v}" for k, v in sorted(b["lean_counts"].items(), key=lambda x: -x[1]))
        lines.append(f"| {b['prompt']} | {b['scheme']} | {_f(b['s_gen_mean'])} | {s} | {b['n']} |")
    lines.append("")

    lines.append("## 6. delivery offset\n")
    if calibration.get("mad_over_sd") is not None:
        lines.append(f"- MAD / between-image sd = **{calibration['mad_over_sd']:.3f}** "
                     f"(MAD {calibration['mad']:.4f}, sd {calibration['sd_between_images']:.4f}, "
                     f"k={calibration['k']})")
    else:
        lines.append("- (no calibration.json)")
    lines.append("")

    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[analyze] wrote {RESULTS_PATH}, {LEAN_PATH}, {LEAN_SUMMARY_PATH}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="run", help="calibrate,run,analyze")
    parser.add_argument("--chunk", type=int, default=0, help="run chunk 0..2")
    parser.add_argument("--n-images", type=int, default=20)
    args = parser.parse_args()
    for phase in [p.strip() for p in args.phase.split(",") if p.strip()]:
        print(f"\n########## phase: {phase} ##########", flush=True)
        if phase == "calibrate":
            phase_calibrate(args.n_images)
        elif phase == "run":
            phase_run(args.chunk)
        elif phase == "analyze":
            phase_analyze()
        else:
            raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    main()
