"""Round-4 pilot driver (docs/bench/10).

Four phases:

* ``items``     -- pick 3 items per bucket (low/mid/high) from the bucket-sampled
  file, writing ``items/pilot_round4.jsonl`` (9 items).
* ``calibrate`` -- delivery-offset calibration: 20 images x 2 schemes, sampled
  across the three buckets (NOT all from one bucket -- round 3 got this wrong
  once), reporting MAD / between-image sd.
* ``run``       -- the 22 generations: arm A (9 items with images), arm B (the
  same 9 items, no image, LVIS category names written into the user turn),
  arm C (no-image baseline, 4 seeds).
* ``analyze``   -- the report numbers + the S1 visualization artifacts
  (``manifest.json``, ``s1viz.json``, and the 27 A-arm jpgs copied to the s1viz
  directory).

Only ``calibrate`` and ``run`` touch the GPU. ``items``/``analyze`` are CPU and
run on the login node; the judge is the standard ``bench.cli judge`` step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bench import registry  # noqa: E402
from bench.adaptors.local_hf import token_scoring, split_probe_id  # noqa: E402
from bench.sample import load_lvis_meta  # noqa: E402
from bench.store import (  # noqa: E402
    RunStore, git_rev, measurement_rev, trial_key,
)
from bench.surfaces.generation import (  # noqa: E402
    AGENTIC_ACK, AGENTIC_OPENER, FILENAMES, FILENAMES_LINE, SYSTEM_AGENTIC, TOOLS,
    ASSISTANT_TURN_1, ASSISTANT_TURN_2, CHAT_USER_TURN_2, SHARE_LINE,
    TASK_PROMPTS,
)
from bench.types import (  # noqa: E402
    BASELINE_ITEM_ID, Conversation, Item, ProbePoint, Trial,
)

OUT_DIR = os.path.join(ROOT_DIR, "runs", "pilot_round4")
PILOT_ITEMS = os.path.join(ROOT_DIR, "items", "pilot_round4.jsonl")
BUCKET_ITEMS = os.path.join(ROOT_DIR, "items", "explore_bucket_v1.jsonl")
S1VIZ_DIR = "/project/jevans/tzhang3/agent-bridge-tmp/uploads/s1viz"

PROBE_ID = "combined_ideology_headwise_linear"
MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
MODEL_FAMILY = "qwen3-vl"
DEFAULT_K = 16
LVIS_CACHE = os.path.join(ROOT_DIR, "items", "_cache", "lvis_image_meta_v2.jsonl")

BUCKETS = ("low", "mid", "high")
BUCKET_ABBREV = {"low": "lo", "mid": "mid", "high": "hi"}
ITEMS_PER_BUCKET = 3
BASELINE_SEEDS = (0, 1, 2, 3)

NEUTRAL_QUESTION = "What stands out to you in these photos?"
S1_QUESTION = TASK_PROMPTS["s1_speech"]


# --------------------------------------------------------------------------- #
# shared conversation builders
# --------------------------------------------------------------------------- #
def _tool_call(name: str, path: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": ""}],
            "tool_calls": [{"type": "function", "function": {"name": name,
                            "arguments": {"path": path}}}]}


def chat_messages(image_path: str, question: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": image_path},
                                          {"type": "text", "text": question}]}]


def agentic_messages(image_path: str, question: str) -> List[Dict[str, Any]]:
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


def _humanize_cat(name: str) -> str:
    """LVIS names read naturally in a sentence: drop '(disambiguator)' and swap
    underscores for spaces, so 'cowboy_hat' -> 'cowboy hat', 'bus_(vehicle)' -> 'bus'."""
    import re
    name = re.sub(r"\(.*\)", "", name)
    return name.replace("_", " ").strip()


def _text_only_messages(category_names: Sequence[str], question: str) -> List[Dict[str, Any]]:
    """Arm B: the chat skeleton verbatim, but the first user turn names the item's
    LVIS categories instead of carrying images (board-buckets control #1)."""
    rendered = ", ".join(_humanize_cat(c) for c in category_names)
    share_text = SHARE_LINE[:-1] + ": " + rendered
    return [
        {"role": "user", "content": [{"type": "text", "text": share_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TURN_1}]},
        {"role": "user", "content": [{"type": "text", "text": CHAT_USER_TURN_2}]},
        {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TURN_2}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]


def _gen_probe_points() -> List[ProbePoint]:
    return [
        ProbePoint(name="s_pre", kind="prefix_end", reduce="last"),
        ProbePoint(name="s_gen", kind="generated_tokens", reduce="mean"),
        ProbePoint(name="s_img", kind="image_tokens", reduce="mean"),
    ]


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #
def phase_items(seed: int = 42) -> None:
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(BUCKET_ITEMS, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_bucket[row["bucket"]].append(row)
    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    for bucket in BUCKETS:
        pool = sorted(by_bucket.get(bucket, []), key=lambda r: r["item_id"])
        rng.shuffle(pool)
        chosen = sorted(pool[:ITEMS_PER_BUCKET], key=lambda r: r["item_id"])
        picked.extend(chosen)
    os.makedirs(os.path.dirname(PILOT_ITEMS), exist_ok=True)
    with open(PILOT_ITEMS, "w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[items] wrote {len(picked)} items -> {PILOT_ITEMS}")
    for bucket in BUCKETS:
        n = sum(1 for r in picked if r["bucket"] == bucket)
        means = [r["covariates"]["image_mean_mean"] for r in picked if r["bucket"] == bucket]
        print(f"  {bucket}: {n} items  image_mean_mean={[f'{m:+.3f}' for m in means]}")


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
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
    runtime = ts.build_probe_runtime(model=model, probe=probe, top_k=DEFAULT_K,
                                     mode="vision", model_family=MODEL_FAMILY)
    module_names = sorted(runtime["module_names"])
    image_ids = sorted(ts.gather_candidate_image_token_ids(processor.tokenizer))

    # Collect n_images image paths spanning all three buckets (round-robin) so
    # the between-image sd reflects the real spread, not one bucket.
    per_bucket: Dict[str, List[str]] = defaultdict(list)
    with open(BUCKET_ITEMS, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            per_bucket[row["bucket"]].extend(row["image_paths"])
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
        a = s_img(chat_messages(path, NEUTRAL_QUESTION))
        b = s_img(agentic_messages(path, NEUTRAL_QUESTION))
        rows.append((a, b))
        print(f"[calibrate {i + 1:2d}/{len(image_paths)}] chat={a:+.4f} agentic={b:+.4f}", flush=True)

    va = np.array([r[0] for r in rows], dtype=float)
    vb = np.array([r[1] for r in rows], dtype=float)
    diff = np.abs(va - vb)
    summary = {
        "n_images": len(rows),
        "k": DEFAULT_K,
        "mean_chat": float(va.mean()),
        "mean_agentic": float(vb.mean()),
        "mean_offset": float((vb - va).mean()),
        "mad": float(diff.mean()),
        "sd_between_images": float(va.std(ddof=1)),
        "mad_over_sd": float(diff.mean() / va.std(ddof=1)),
    }
    with open(os.path.join(OUT_DIR, "calibration.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _plan(surface, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the 22 trials as dicts (built before the adaptor loads, so the plan
    count printed is the count actually run)."""
    s1 = surface
    plan: List[Dict[str, Any]] = []

    # Arm A: 9 items, with images (condition C).
    for row in items:
        item = Item.from_dict(row)
        trial = s1.build(item, "C", {"scheme": "chat"})
        plan.append({"trial": trial, "arm": "A", "condition": "C",
                     "item_id": item.item_id, "bucket": row["bucket"],
                     "stratum": row["stratum"], "image_scores": item.image_scores,
                     "image_mean": item.image_mean, "covariates": item.covariates,
                     "split": item.split, "is_baseline": False, "seed": 42})

    # Arm B: the same 9 items, text-only (category names, no image).
    for row in items:
        item = Item.from_dict(row)
        cats = (item.covariates or {}).get("categories") or []
        messages = _text_only_messages(cats, S1_QUESTION)
        trial = Trial(
            surface="s1_speech", item_id=item.item_id, condition="B",
            conversation=Conversation(messages=messages, images=[]),
            candidates=[], probe_points=_gen_probe_points(),
            max_new_tokens=s1.max_new_tokens, variant={"scheme": "chat", "text_only": True},
            meta={"family": "generation", "scheme": "chat", "question": S1_QUESTION,
                  "tools": None, "prefix_n_messages": len(messages) - 1,
                  "condition_desc": "text-only: LVIS category names, no images",
                  "n_images": 0, "item_invariant": False, "judge": "s1_speech",
                  "category_names": cats},
        )
        plan.append({"trial": trial, "arm": "B", "condition": "B",
                     "item_id": item.item_id, "bucket": row["bucket"],
                     "stratum": row["stratum"], "image_scores": item.image_scores,
                     "image_mean": item.image_mean, "covariates": item.covariates,
                     "split": item.split, "is_baseline": False, "seed": 42})

    # Arm C: no-photo baseline -- the S1 question alone, no photo mention at all
    # (stricter than round-3's condition E, which still said "these are some
    # photos I took recently"). 4 seeds; greedy decoding means the four are
    # byte-identical, but the records are distinct (seed is in the trial_key).
    for seed in BASELINE_SEEDS:
        messages = [{"role": "user", "content": [{"type": "text", "text": S1_QUESTION}]}]
        trial = Trial(
            surface="s1_speech", item_id=BASELINE_ITEM_ID, condition="E",
            conversation=Conversation(messages=messages, images=[]),
            candidates=[], probe_points=_gen_probe_points(),
            max_new_tokens=s1.max_new_tokens, variant={"scheme": "chat"},
            meta={"family": "generation", "scheme": "chat", "question": S1_QUESTION,
                  "tools": None, "prefix_n_messages": len(messages),
                  "condition_desc": "no-photo baseline: question only, no framing",
                  "n_images": 0, "item_invariant": True, "judge": "s1_speech"},
        )
        plan.append({"trial": trial, "arm": "C", "condition": "E",
                     "item_id": BASELINE_ITEM_ID, "bucket": None,
                     "stratum": None, "image_scores": [], "image_mean": None,
                     "covariates": {}, "split": "explore", "is_baseline": True,
                     "seed": seed})

    return plan


def phase_run() -> None:
    import torch

    registry.load_all()
    from bench.adaptors.local_hf import LocalHFAdaptor

    s1 = registry.get_surface("s1_speech")()

    with open(PILOT_ITEMS, encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]

    plan = _plan(s1, items)
    print(f"[run] {len(items)} items -> {len(plan)} trials "
          f"(A={sum(1 for p in plan if p['arm'] == 'A')}, "
          f"B={sum(1 for p in plan if p['arm'] == 'B')}, "
          f"C={sum(1 for p in plan if p['arm'] == 'C')})")

    adaptor = LocalHFAdaptor(model=MODEL_PATH, probe=PROBE_ID, top_k=DEFAULT_K, seed=42)
    probe_weights = adaptor.describe().get("probe_weights")
    rev = measurement_rev(ROOT_DIR, extra_files=[probe_weights] if probe_weights else [],
                          note=f"top_k={DEFAULT_K}")
    code_rev = git_rev(ROOT_DIR)
    print(f"[run] code_rev={code_rev}  measurement_rev={rev}")

    store = RunStore(run_dir=OUT_DIR, conversations_dir=os.path.join(ROOT_DIR, "conversations"))
    model_id = adaptor.model
    print(f"[run] resume: {store.n_done} trials already in {store.trials_path}")

    import time as _time
    started = _time.time()

    adaptor.setup()

    n_new = n_skip = n_err = 0
    try:
        for entry in plan:
            trial = entry["trial"]
            key = trial_key("s1_speech", entry["item_id"], entry["condition"],
                            trial.variant, adaptor.name, model_id, entry["seed"], rev)
            if store.has(key):
                n_skip += 1
                continue
            conversation_sha = store.put_conversation(trial.conversation)
            torch.manual_seed(entry["seed"])
            response = adaptor.run(trial)
            if response.error:
                n_err += 1
                print(f"  ! {entry['arm']}/{entry['item_id']}: {response.error}", file=sys.stderr)
            record = {
                "trial_key": key,
                "run_id": os.path.basename(OUT_DIR.rstrip("/")),
                "code_rev": code_rev,
                "measurement_rev": rev,
                "surface": "s1_speech",
                "surface_family": "generation",
                "condition": entry["condition"],
                "arm": entry["arm"],
                "variant": trial.variant,
                "item_id": entry["item_id"],
                "is_baseline": entry["is_baseline"],
                "stratum": entry["stratum"],
                "bucket": entry["bucket"],
                "primary_iv": "bucket",
                "split": entry["split"],
                "images": trial.conversation.images,
                "image_scores": entry["image_scores"],
                "image_mean": entry["image_mean"],
                "adaptor": adaptor.name,
                "model": model_id,
                "seed": entry["seed"],
                "conversation_sha": conversation_sha,
                "response": response.to_dict(),
                "probe": response.probe,
                "outcome": None,
                "needs_judge": None,
                "judge": None,
                "timing": {"ms": response.timing_ms},
                "cost_usd": response.cost_usd,
                "error": response.error,
                "covariates": entry["covariates"],
            }
            store.append(record)
            n_new += 1
            print(f"  [{n_new}/{len(plan)}] arm={entry['arm']} {entry['item_id']} "
                  f"cond={entry['condition']} seed={entry['seed']} "
                  f"s_pre={response.probe['s_pre'] if response.probe else None:+.4f} "
                  f"s_gen={response.probe['s_gen'] if response.probe else None:+.4f}",
                  flush=True)
    finally:
        store.write_manifest(sys.argv, code_rev, adaptor.describe(), started,
                             finished_at=_time.time(),
                             extra={"n_new": n_new, "n_skipped": n_skip, "n_errors": n_err},
                             measurement_rev=rev)
        store.close()
        adaptor.teardown()

    print(f"\n[run] new={n_new} skipped={n_skip} errors={n_err}")
    print(f"[run] {store.trials_path}")


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def _read_trials() -> List[Dict[str, Any]]:
    path = os.path.join(OUT_DIR, "trials.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_judges() -> List[Dict[str, Any]]:
    path = os.path.join(OUT_DIR, "judges.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _f(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:+.4f}"


def phase_analyze() -> None:
    trials = _read_trials()
    judges = _read_judges()
    if not trials:
        print("no trials; run the GPU phase first", file=sys.stderr)
        raise SystemExit(1)

    judge_by_key = {j["trial_key"]: j for j in judges}

    by_item: Dict[str, Dict[str, Any]] = {}
    baselines: List[Dict[str, Any]] = []
    for r in trials:
        if r.get("arm") == "C":
            baselines.append(r)
            continue
        if r.get("arm") in ("A", "B"):
            by_item.setdefault(r["item_id"], {})[r["arm"]] = r

    meta = load_lvis_meta(LVIS_CACHE)
    item_rows = [json.loads(line) for line in open(PILOT_ITEMS, encoding="utf-8") if line.strip()]

    lines: List[str] = []
    lines.append("# round-4 buckets + text-only pilot analysis\n")

    # --- arm A: s_gen per bucket, and A vs B difference ----------------------
    lines.append("## s_gen per bucket (arm A) and A vs B (text-only) difference\n")
    lines.append("| bucket | A s_gen | B s_gen | A - B | n |")
    lines.append("|---|---|---|---|---|")
    for bucket in BUCKETS:
        a = [r for r in by_item.values() if r.get("A", {}).get("bucket") == bucket and r["A"].get("probe")]
        b = [r for r in by_item.values() if r.get("B", {}).get("bucket") == bucket and r["B"].get("probe")]
        a_gen = _mean([r["A"]["probe"]["s_gen"] for r in a if r["A"]["probe"].get("s_gen") is not None])
        b_gen = _mean([r["B"]["probe"]["s_gen"] for r in b if r["B"]["probe"].get("s_gen") is not None])
        diff = (a_gen - b_gen) if (a_gen is not None and b_gen is not None) else None
        lines.append(f"| {bucket} | {_f(a_gen)} | {_f(b_gen)} | {_f(diff)} | {len(a)} |")
    lines.append("")

    # --- arm A: s_pre per bucket --------------------------------------------
    lines.append("## s_pre per bucket (arm A)\n")
    lines.append("| bucket | s_pre | n |")
    lines.append("|---|---|---|")
    for bucket in BUCKETS:
        a = [r["A"] for r in by_item.values()
             if r.get("A", {}).get("bucket") == bucket and r["A"].get("probe")]
        pre = _mean([r["probe"]["s_pre"] for r in a if r["probe"].get("s_pre") is not None])
        lines.append(f"| {bucket} | {_f(pre)} | {len(a)} |")
    lines.append("")

    # --- arm C baseline ------------------------------------------------------
    lines.append("## arm C baseline (no image, 4 seeds)\n")
    c_gen = [r["probe"]["s_gen"] for r in baselines if r.get("probe") and r["probe"].get("s_gen") is not None]
    c_pre = [r["probe"]["s_pre"] for r in baselines if r.get("probe") and r["probe"].get("s_pre") is not None]
    lines.append(f"- s_gen: mean **{_f(_mean(c_gen))}**  values: {[f'{v:+.4f}' for v in c_gen]}")
    lines.append(f"- s_pre: mean **{_f(_mean(c_pre))}**  values: {[f'{v:+.4f}' for v in c_pre]}")
    lines.append("")

    # --- calibration ---------------------------------------------------------
    calib_path = os.path.join(OUT_DIR, "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path, encoding="utf-8") as handle:
            calib = json.load(handle)
        lines.append("## delivery offset (MAD / between-image sd)\n")
        lines.append(f"- MAD / sd = **{calib['mad_over_sd']:.3f}**  "
                     f"(mean offset {calib['mean_offset']:+.4f}, MAD {calib['mad']:.4f}, "
                     f"sd {calib['sd_between_images']:.4f}, n={calib['n_images']})")
        lines.append("")

    # --- judge distributions -------------------------------------------------
    if judge_by_key:
        lines.append("## judge field distributions (s1_speech, 8 fields)\n")
        from bench.judges import judge_specs
        spec = judge_specs()["s1_speech"]
        by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in trials:
            j = judge_by_key.get(r["trial_key"])
            if j and "labels" in j:
                by_arm[r.get("arm")].append(j)
        label_fields = [f for f in spec.fields if f not in ("named_attributes",)]
        for arm in ("A", "B", "C"):
            subs = by_arm.get(arm, [])
            lines.append(f"### arm {arm}  (n={len(subs)})")
            for field in label_fields:
                vals = [j["labels"].get(field) for j in subs if j.get("labels", {}).get(field) is not None]
                if not vals:
                    continue
                dist = Counter(str(v) for v in vals)
                rendered = "  ".join(f"{k}:{dist[k]}" for k in sorted(dist, key=lambda x: -dist[x]))
                lines.append(f"- `{field}`: {rendered}")
            lines.append("")
    else:
        lines.append("## judge\n(no judges.jsonl -- run `bench.cli judge`)\n")

    report_path = os.path.join(OUT_DIR, "RESULTS.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[analyze] wrote {report_path}")

    # --- s1viz artifacts -----------------------------------------------------
    _write_s1viz(item_rows, by_item, baselines, meta, judge_by_key)


def _write_s1viz(item_rows, by_item, baselines, meta, judge_by_key) -> None:
    os.makedirs(S1VIZ_DIR, exist_ok=True)
    item_rows = sorted(item_rows, key=lambda r: r["item_id"])
    bucket_order: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in item_rows:
        bucket_order[r["bucket"]].append(r)
    for b in BUCKETS:
        bucket_order[b] = sorted(bucket_order[b], key=lambda r: r["item_id"])

    manifest: List[Dict[str, Any]] = []
    s1viz_items: List[Dict[str, Any]] = []

    for bucket in BUCKETS:
        for seq, row in enumerate(bucket_order[bucket], start=1):
            item_id = row["item_id"]
            cov = row["covariates"]
            coco_ids = cov["coco_ids"]
            image_scores = row["image_scores"]
            cats_merged = cov["categories"]

            per_image_cats: List[List[str]] = []
            per_image_n: List[int] = []
            per_image_inst: List[int] = []
            for cid in coco_ids:
                info = meta.get(cid) or {"categories": [], "n_objects": 0}
                per_image_cats.append(list(info["categories"]))
                per_image_n.append(len(info["categories"]))
                per_image_inst.append(int(info["n_objects"]))

            for img_idx, cid in enumerate(coco_ids, start=1):
                manifest.append({
                    "file": f"{bucket}_{seq}_{img_idx}.jpg",
                    "bucket": bucket,
                    "item_id": item_id,
                    "img_index": img_idx,
                    "coco_id": cid,
                    "image_mean": image_scores[img_idx - 1],
                    "categories": per_image_cats[img_idx - 1],
                    "n_objects": per_image_n[img_idx - 1],
                    "n_instances": per_image_inst[img_idx - 1],
                })

            arms: Dict[str, Any] = {}
            for arm in ("A", "B"):
                rec = by_item.get(item_id, {}).get(arm)
                if not rec:
                    continue
                j = judge_by_key.get(rec["trial_key"], {})
                arms[arm] = {
                    "s_pre": rec["probe"]["s_pre"],
                    "s_gen": rec["probe"]["s_gen"],
                    "s_gen_first25": rec["probe"].get("s_gen_first25"),
                    "s_gen_last25": rec["probe"].get("s_gen_last25"),
                    "judge_labels": j.get("labels"),
                    "text": (rec.get("response") or {}).get("text"),
                }

            s1viz_items.append({
                "item_id": item_id,
                "bucket": bucket,
                "image_mean": row["covariates"].get("image_mean_mean"),
                "image_means": image_scores,
                "categories": cats_merged,
                "n_objects": _mean(per_image_n),
                "n_instances": _mean(per_image_inst),
                "coco_ids": coco_ids,
                "arms": arms,
            })

    # arm C baseline (item-invariant, reported once)
    baseline_runs: List[Dict[str, Any]] = []
    for r in baselines:
        j = judge_by_key.get(r["trial_key"], {})
        baseline_runs.append({
            "seed": r["seed"],
            "s_pre": r["probe"]["s_pre"],
            "s_gen": r["probe"]["s_gen"],
            "s_gen_first25": r["probe"].get("s_gen_first25"),
            "s_gen_last25": r["probe"].get("s_gen_last25"),
            "judge_labels": j.get("labels"),
            "text": (r.get("response") or {}).get("text"),
        })
    baseline_mean_s_pre = _mean([r["s_pre"] for r in baseline_runs if r["s_pre"] is not None])
    baseline_mean_s_gen = _mean([r["s_gen"] for r in baseline_runs if r["s_gen"] is not None])

    s1viz = {
        "items": s1viz_items,
        "baseline": {
            "arm": "C",
            "runs": baseline_runs,
            "mean_s_pre": baseline_mean_s_pre,
            "mean_s_gen": baseline_mean_s_gen,
        },
    }

    # Copy the 27 A-arm images (the model sees the _resized_images_800 version).
    copied = 0
    for bucket in BUCKETS:
        for seq, row in enumerate(bucket_order[bucket], start=1):
            for img_idx, src in enumerate(row["image_paths"], start=1):
                dst = os.path.join(S1VIZ_DIR, f"{bucket}_{seq}_{img_idx}.jpg")
                shutil.copyfile(src, dst)
                copied += 1

    manifest_path = os.path.join(S1VIZ_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    s1viz_path = os.path.join(S1VIZ_DIR, "s1viz.json")
    with open(s1viz_path, "w", encoding="utf-8") as handle:
        json.dump(s1viz, handle, indent=1, sort_keys=True)

    print(f"\n[s1viz] copied {copied} images -> {S1VIZ_DIR}")
    print(f"[s1viz] manifest.json: {len(manifest)} entries -> {manifest_path}")
    print(f"[s1viz] s1viz.json: {len(s1viz_items)} items -> {s1viz_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", default="items", help="items,calibrate,run,analyze")
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for phase in [p.strip() for p in args.phases.split(",") if p.strip()]:
        print(f"\n########## phase: {phase} ##########", flush=True)
        if phase == "items":
            phase_items(args.seed)
        elif phase == "calibrate":
            phase_calibrate(args.n_images)
        elif phase == "run":
            phase_run()
        elif phase == "analyze":
            phase_analyze()
        else:
            raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    main()
