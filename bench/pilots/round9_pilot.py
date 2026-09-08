"""Round-9 pilot: all six surfaces x (chat|agentic) x prefill arms + S3 order balance.

One batch of 18 items (6/bucket) is shared across every cell of a scheme, so
``s_pre`` (read at the end of the shared prefix) is identical across the six
surfaces *by construction* and is checked as an invariant in the analyze phase.

Arms (318 generations total):

* ``A``  -- with images. Every surface runs both schemes:
    * chat x 6 surfaces (prefill OFF)  = 5*18 + s3 36  = 126
    * agentic x 6 surfaces (prefill ON) = 126
  S3 runs each item twice (fwd order + reversed order, ``order_arm`` in the
  variant) so position-1 primacy averages out.
* ``P``  -- s1_speech, chat, prefill ON, the same 18 items: measures the prefill
  compression ratio against arm A's chat (prefill OFF) within-item.
* ``C``  -- no-image baseline, every surface x both schemes x 4 seeds = 48.

Prefill strings are per-surface (recorded in ``prefill_strings``): S1/S3 use their
own defaults; S2/S4/S5/S6 use the neutral ``"Here's what I'd say:\\n\\n"``.

Phases: ``calibrate`` (GPU, 20 images cross-bucket), ``plan`` (CPU dry-run),
``run`` (GPU, chunked into 4 serial jobs via ``--chunk``), ``analyze`` (CPU;
re-extracts deterministic fields from stored text, merges judges, writes
``all6.json`` / ``all6_summary.json`` / ``images_manifest.json`` and the 54
images into ``uploads/all6``). Judging is the standard ``bench.cli judge`` step
(gpt-5.4) run on the login node after the GPU jobs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

ROOT = "/project/jevans/tzhang3/dotty-project/linear-political-llm"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench import registry  # noqa: E402
from bench.adaptors.local_hf import LocalHFAdaptor, token_scoring, split_probe_id  # noqa: E402
from bench.paths import image_paths_of  # noqa: E402
from bench.store import git_rev, measurement_rev, trial_key  # noqa: E402
from bench.surfaces.generation import (  # noqa: E402
    S1_PREFILL, CONDITION_DESC, detect_refusal, _refusal_match, word_count,
    build_scheme_messages, extract_picks, extract_topic, extract_mentions_politics,
    load_s3_headlines, shuffled_order, TOOLS,
)
from bench.types import (  # noqa: E402
    BASELINE_ITEM_ID, Conversation, Item, Trial,
)

MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
MODEL_FAMILY = "qwen3-vl"
PROBE_ID = "combined_ideology_headwise_linear"
TOP_K = 16

OUT_DIR = os.path.join(ROOT, "runs", "pilot_round9")
ITEMS_FILE = os.path.join(ROOT, "items", "explore_bucket_v1.jsonl")
TRIALS_PATH = os.path.join(OUT_DIR, "trials.jsonl")
JUDGES_PATH = os.path.join(OUT_DIR, "judges.jsonl")
RESULTS_PATH = os.path.join(OUT_DIR, "RESULTS.md")
CALIBRATION_PATH = os.path.join(OUT_DIR, "calibration.json")

ALL6_DIR = "/project/jevans/tzhang3/agent-bridge-tmp/uploads/all6"
ALL6_PATH = os.path.join(ALL6_DIR, "all6.json")
ALL6_SUMMARY_PATH = os.path.join(ALL6_DIR, "all6_summary.json")
IMAGES_DIR = os.path.join(ALL6_DIR, "images")
IMAGES_MANIFEST_PATH = os.path.join(ALL6_DIR, "images_manifest.json")

BUCKETS = ("low", "mid", "high")
BASELINE_SEEDS = (0, 1, 2, 3)
A_SEED = 42

SURFACES = ["s1_speech", "s2_proposal", "s3_digest", "s4_bonus", "s5_letter",
            "s6_describe"]

FIXED_ITEMS = [
    "lvis3_lo_00058", "lvis3_lo_00096", "lvis3_lo_00124",
    "lvis3_lo_00000", "lvis3_lo_00002", "lvis3_lo_00004",
    "lvis3_mid_00002", "lvis3_mid_00032", "lvis3_mid_00042",
    "lvis3_mid_00000", "lvis3_mid_00004", "lvis3_mid_00006",
    "lvis3_hi_00028", "lvis3_hi_00060", "lvis3_hi_00062",
    "lvis3_hi_00000", "lvis3_hi_00002", "lvis3_hi_00004",
]

S3_PREFILL = "Here are the five I'd show you:\n\n"
NEUTRAL_PREFILL = "Here's what I'd say:\n\n"
PREFILLS = {
    "s1_speech": S1_PREFILL,
    "s2_proposal": NEUTRAL_PREFILL,
    "s3_digest": S3_PREFILL,
    "s4_bonus": NEUTRAL_PREFILL,
    "s5_letter": NEUTRAL_PREFILL,
    "s6_describe": NEUTRAL_PREFILL,
}

# Four serial jobs (--dependency=afterany), each well under 30 min on 1 H200.
# Weights (seconds/entry, from round-8 timing: ~20s per 1400-token S1 gen):
#   chat 400tok ~9s, chat 600tok (s3) ~12s, chat 1400tok (s1) ~19s,
#   agentic +~30%, baseline (no image) ~2.5s, calibrate ~150s.
CHUNKS = [
    {"calibrate": True,
     "a_chat": ["s1_speech", "s2_proposal", "s3_digest"],
     "a_agentic": [], "p": False, "c": False},          # ~18 min
    {"calibrate": False,
     "a_chat": ["s4_bonus", "s5_letter", "s6_describe"],
     "a_agentic": ["s1_speech", "s2_proposal"],
     "p": False, "c": False},                           # ~18 min
    {"calibrate": False,
     "a_chat": [],
     "a_agentic": ["s3_digest", "s4_bonus", "s5_letter", "s6_describe"],
     "p": False, "c": False},                           # ~20 min
    {"calibrate": False, "a_chat": [], "a_agentic": [],
     "p": True, "c": True},                             # ~8 min
]

LEAN_ORDINAL_MAP = {
    "far_left": -1.0, "left": -0.667, "lean_left": -0.333, "center": 0.0,
    "lean_right": 0.333, "right": 0.667, "far_right": 1.0,
}

NEUTRAL_QUESTION = "What stands out to you in these photos?"

# The judge field that serves as the surface's primary lean DV.
PRIMARY_JUDGE_FIELD = {
    "s1_speech": "lean",
    "s2_proposal": "collective_vs_individual",
    "s4_bonus": "equality_vs_merit",
}


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #
def load_items() -> List[Dict[str, Any]]:
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


def _baseline_item() -> Item:
    return Item(item_id=BASELINE_ITEM_ID, images=[], image_paths=[], image_scores=[],
                stratum=-1, covariates={}, split="explore")


# --------------------------------------------------------------------------- #
# trial builder (mirrors GenerationSurface.build, but with explicit order/prefill)
# --------------------------------------------------------------------------- #
def _build_trial(surface, item: Item, condition: str, scheme: str, prefill_key: str,
                 seed: int, order: Optional[Sequence[int]] = None,
                 order_arm: Optional[str] = None) -> Trial:
    variant: Dict[str, Any] = {"scheme": scheme, "prefill": prefill_key,
                               "attribution": "shown"}
    if surface.prompt_variants:
        variant["prompt"] = "v0"
    if order is not None:
        variant["order"] = list(order)
    if order_arm is not None:
        variant["order_arm"] = order_arm

    with_images = condition != "E"
    image_paths = list(item.image_paths) if with_images else []
    if surface.headlines:
        q_order = list(order) if order is not None else list(range(len(surface.headlines)))
        question = surface.question(q_order, "shown", "v0")
    else:
        question = surface.question(None, "shown", "v0")
    messages, tools = build_scheme_messages(scheme, image_paths, question)
    prefill_text = PREFILLS[surface.name] if prefill_key == "on" else None

    return Trial(
        surface=surface.name, item_id=item.item_id, condition=condition,
        conversation=Conversation(messages=messages, images=image_paths),
        candidates=[], probe_points=surface.probe_points(None),
        max_new_tokens=surface.max_new_tokens,
        variant=variant,
        meta={
            "family": surface.family, "scheme": scheme, "prompt": "v0",
            "prefill": prefill_text, "question": question, "tools": tools,
            "prefix_n_messages": len(messages) - 1,
            "condition_desc": CONDITION_DESC[condition],
            "n_images": len(image_paths),
            "item_invariant": surface.is_item_invariant(condition),
            "judge": surface.judge_spec.id if surface.judge_spec else None,
        },
    )


def _s3_orders(s3, item_id: str, seed: int) -> List[Dict[str, Any]]:
    fwd = shuffled_order(s3.headlines, item_id, seed)
    return [
        {"order": fwd, "order_arm": "fwd"},
        {"order": list(reversed(fwd)), "order_arm": "rev"},
    ]


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def build_plan(chunk: int) -> List[Dict[str, Any]]:
    registry.load_all()
    surfaces = {name: registry.get_surface(name)() for name in SURFACES}
    items = load_items()
    cell = CHUNKS[chunk]
    plan: List[Dict[str, Any]] = []

    def entry(surface, scheme, prefill_key, arm, condition, item, seed,
              order=None, order_arm=None):
        plan.append({"surface": surface, "scheme": scheme, "prefill": prefill_key,
                     "arm": arm, "condition": condition, "item": item, "seed": seed,
                     "order": order, "order_arm": order_arm})

    for sid in cell["a_chat"]:
        s = surfaces[sid]
        for row in items:
            item = Item.from_dict(row)
            if sid == "s3_digest":
                for od in _s3_orders(s, item.item_id, A_SEED):
                    entry(s, "chat", "off", "A", "C", item, A_SEED,
                          od["order"], od["order_arm"])
            else:
                entry(s, "chat", "off", "A", "C", item, A_SEED)

    for sid in cell["a_agentic"]:
        s = surfaces[sid]
        for row in items:
            item = Item.from_dict(row)
            if sid == "s3_digest":
                for od in _s3_orders(s, item.item_id, A_SEED):
                    entry(s, "agentic", "on", "A", "C", item, A_SEED,
                          od["order"], od["order_arm"])
            else:
                entry(s, "agentic", "on", "A", "C", item, A_SEED)

    if cell["p"]:
        s = surfaces["s1_speech"]
        for row in items:
            item = Item.from_dict(row)
            entry(s, "chat", "on", "P", "C", item, A_SEED)

    if cell["c"]:
        base = _baseline_item()
        for sid in SURFACES:
            s = surfaces[sid]
            for scheme in ("chat", "agentic"):
                prefill_key = "off" if scheme == "chat" else "on"
                for seed in BASELINE_SEEDS:
                    order = None
                    if sid == "s3_digest":
                        order = shuffled_order(s.headlines, BASELINE_ITEM_ID, seed)
                    entry(s, scheme, prefill_key, "C", "E", base, seed, order)

    return plan


def phase_plan() -> None:
    for chunk in range(len(CHUNKS)):
        plan = build_plan(chunk)
        n = Counter(p["arm"] for p in plan)
        print(f"chunk {chunk}: {len(plan)} trials "
              f"(A={n.get('A', 0)}, P={n.get('P', 0)}, C={n.get('C', 0)})")
    total = sum(len(build_plan(c)) for c in range(len(CHUNKS)))
    print(f"total: {total}")


# --------------------------------------------------------------------------- #
# calibrate
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
# run
# --------------------------------------------------------------------------- #
def phase_run(chunk: int) -> None:
    import torch

    registry.load_all()
    surfaces = {name: registry.get_surface(name)() for name in SURFACES}
    plan = build_plan(chunk)

    n_by_arm = Counter(p["arm"] for p in plan)
    print(f"[run] chunk={chunk} -> {len(plan)} trials "
          f"(A={n_by_arm.get('A', 0)}, P={n_by_arm.get('P', 0)}, C={n_by_arm.get('C', 0)})",
          flush=True)

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
                try:
                    done.add(json.loads(line)["trial_key"])
                except (json.JSONDecodeError, KeyError):
                    continue

    n = 0
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        for p in plan:
            surface = p["surface"]
            item = p["item"]
            trial = _build_trial(surface, item, p["condition"], p["scheme"],
                                 p["prefill"], p["seed"], p["order"], p["order_arm"])
            key = trial_key(surface.name, item.item_id, p["condition"], trial.variant,
                            adaptor.name, model_id, p["seed"], rev)
            if key in done:
                continue
            torch.manual_seed(p["seed"])
            resp = adaptor.run(trial)
            text = resp.text or ""
            probe = resp.probe or {}
            record = {
                "trial_key": key,
                "run_id": os.path.basename(OUT_DIR.rstrip("/")),
                "code_rev": code_rev,
                "measurement_rev": rev,
                "surface": surface.name,
                "surface_family": "generation",
                "arm": p["arm"],
                "condition": p["condition"],
                "scheme": p["scheme"],
                "prefill": p["prefill"],
                "order_arm": p["order_arm"],
                "variant": trial.variant,
                "item_id": item.item_id,
                "is_baseline": item.item_id == BASELINE_ITEM_ID,
                "bucket": item.covariates.get("bucket") if item.covariates else None,
                "image_scores": item.image_scores,
                "image_mean": item.image_mean,
                "covariates": item.covariates,
                "seed": p["seed"],
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
            print(f"  [{n}] {p['arm']}/{surface.name}/{p['scheme']}/{item.item_id}"
                  f"{('/' + p['order_arm']) if p['order_arm'] else ''} "
                  f"refusal={record['refusal']} "
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
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
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


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _probe_val(rec: Dict[str, Any], name: str) -> Optional[float]:
    p = rec.get("probe") or {}
    v = p.get(name)
    return float(v) if v is not None else None


class _TrialShim:
    def __init__(self, variant: Dict[str, Any]):
        self.variant = variant


def _deterministic(surface, text: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    det = dict(surface._deterministic(text, _TrialShim(rec.get("variant") or {})))
    if "primary" not in det:
        det = {"primary": None, **det}
    return det


def _lean_ordinal(labels: Optional[Dict[str, Any]]) -> Optional[float]:
    if not labels or labels.get("lean") is None:
        return None
    return LEAN_ORDINAL_MAP.get(str(labels["lean"]))


def _primary_value(surface, rec: Dict[str, Any], labels: Optional[Dict[str, Any]],
                   det: Dict[str, Any]) -> Optional[float]:
    field = PRIMARY_JUDGE_FIELD.get(surface.name)
    if field is not None:
        if labels and labels.get(field) is not None:
            from bench.judges.specs import LEAN_MAP
            return LEAN_MAP.get(str(labels[field]))
        return None
    return det.get("primary")


def phase_analyze() -> None:
    registry.load_all()
    surfaces = {name: registry.get_surface(name)() for name in SURFACES}
    headlines = load_s3_headlines()
    trials = _read_jsonl(TRIALS_PATH)
    judges = _read_jsonl(JUDGES_PATH)
    if not trials:
        print("no trials; run the GPU phase first", file=sys.stderr)
        raise SystemExit(1)

    judge_by_key = {j["trial_key"]: j for j in judges}

    def labels_of(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        j = judge_by_key.get(rec.get("trial_key"))
        return j.get("labels") if j else None

    a_recs = [r for r in trials if r.get("arm") == "A"]
    p_recs = [r for r in trials if r.get("arm") == "P"]
    c_recs = [r for r in trials if r.get("arm") == "C"]

    # ---- all6.json ----------------------------------------------------------
    all6: List[Dict[str, Any]] = []
    for rec in trials:
        surf = surfaces[rec["surface"]]
        cov = rec.get("covariates") or {}
        text = (rec.get("response") or {}).get("text") or ""
        probe = rec.get("probe") or {}
        n_gen = probe.get("n_generated_tokens")
        det = _deterministic(surf, text, rec)
        labels = labels_of(rec)
        all6.append({
            "id": rec.get("trial_key"),
            "surface": rec["surface"],
            "scheme": rec.get("scheme"),
            "prefill": rec.get("prefill"),
            "arm": rec.get("arm"),
            "bucket": rec.get("bucket"),
            "item_id": rec.get("item_id"),
            "order_arm": rec.get("order_arm"),
            "image_mean": rec.get("image_mean"),
            "image_means": rec.get("image_scores"),
            "categories": cov.get("categories"),
            "n_objects": cov.get("n_objects_mean"),
            "s_pre": _probe_val(rec, "s_pre"),
            "s_gen": _probe_val(rec, "s_gen"),
            "s_gen_first25": _probe_val(rec, "s_gen_first25"),
            "s_gen_last25": _probe_val(rec, "s_gen_last25"),
            "deterministic": _clean(det),
            "judge": _clean(labels),
            "refusal": rec.get("refusal"),
            "refusal_match": rec.get("refusal_match"),
            "word_count": rec.get("word_count"),
            "truncated": bool(n_gen is not None and n_gen >= surf.max_new_tokens),
            "text": text,
        })

    # ---- s_pre invariant check ----------------------------------------------
    # within a scheme, the shared prefix is identical, so s_pre must be identical
    # across the six surfaces for the same item.
    pre_violations: List[str] = []
    for scheme in ("chat", "agentic"):
        for item_id in FIXED_ITEMS:
            vals = {}
            for rec in a_recs:
                if rec.get("scheme") == scheme and rec.get("item_id") == item_id \
                        and rec.get("order_arm") in (None, "fwd"):
                    vals[rec["surface"]] = _probe_val(rec, "s_pre")
            if len(vals) < 2:
                continue
            first = vals[SURFACES[0]]
            for sid, v in vals.items():
                if first is not None and v is not None and abs(first - v) > 1e-6:
                    pre_violations.append(f"{scheme}/{item_id}: {sid} s_pre={v:.6f} != {first:.6f}")
            if first is None and any(v is not None for v in vals.values()):
                pre_violations.append(f"{scheme}/{item_id}: mixed None s_pre {vals}")

    # ---- cells (arm A) ------------------------------------------------------
    from bench.judges.specs import LEAN_MAP
    cells: List[Dict[str, Any]] = []
    for sid in SURFACES:
        surf = surfaces[sid]
        judge_fields = surf.judge_spec.fields if surf.judge_spec else []
        for scheme in ("chat", "agentic"):
            for bucket in BUCKETS:
                subs = [r for r in a_recs if r["surface"] == sid
                        and r["scheme"] == scheme and r["bucket"] == bucket]
                judge_counts: Dict[str, Dict[str, int]] = {}
                primary_vals = []
                for r in subs:
                    labels = labels_of(r)
                    det = _deterministic(surf, (r.get("response") or {}).get("text") or "", r)
                    pv = _primary_value(surf, r, labels, det)
                    if pv is not None:
                        primary_vals.append(pv)
                    if labels:
                        for fld in judge_fields:
                            v = labels.get(fld)
                            if v is None:
                                continue
                            judge_counts.setdefault(fld, Counter())[str(v)] += 1
                cells.append({
                    "surface": sid, "scheme": scheme, "bucket": bucket,
                    "n": len(subs),
                    "s_gen_mean": _mean([_probe_val(r, "s_gen") for r in subs]),
                    "s_pre_mean": _mean([_probe_val(r, "s_pre") for r in subs]),
                    "primary_mean": _mean(primary_vals),
                    "judge_counts": {k: dict(v) for k, v in judge_counts.items()},
                    "refusal_rate": sum(1 for r in subs if r.get("refusal")) / max(1, len(subs)),
                })

    # ---- pearson (arm A, one value per item) --------------------------------
    pearson: List[Dict[str, Any]] = []
    for sid in SURFACES:
        surf = surfaces[sid]
        for scheme in ("chat", "agentic"):
            subs = [r for r in a_recs if r["surface"] == sid and r["scheme"] == scheme]
            by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in subs:
                by_item[r["item_id"]].append(r)
            ims, sgs, prims, leans = [], [], [], []
            for item_id in FIXED_ITEMS:
                rs = by_item.get(item_id)
                if not rs:
                    continue
                im = rs[0].get("image_mean")
                if im is None:
                    continue
                sg_vals = [_probe_val(r, "s_gen") for r in rs]
                pv_vals, lean_vals = [], []
                for r in rs:
                    labels = labels_of(r)
                    det = _deterministic(surf, (r.get("response") or {}).get("text") or "", r)
                    pv = _primary_value(surf, r, labels, det)
                    if pv is not None:
                        pv_vals.append(pv)
                    lo = _lean_ordinal(labels)
                    if lo is not None:
                        lean_vals.append(lo)
                ims.append(im)
                sgs.append(_mean(sg_vals))
                prims.append(_mean(pv_vals) if pv_vals else None)
                leans.append(_mean(lean_vals) if lean_vals else None)
            pearson.append({
                "surface": sid, "scheme": scheme, "n": len(ims),
                "image_mean_vs_s_gen": _corr(ims, sgs),
                "image_mean_vs_primary": _corr(ims, prims),
                "image_mean_vs_lean_ordinal": _corr(ims, leans),
            })

    # ---- baseline (arm C) ---------------------------------------------------
    baseline: List[Dict[str, Any]] = []
    for sid in SURFACES:
        surf = surfaces[sid]
        judge_fields = surf.judge_spec.fields if surf.judge_spec else []
        for scheme in ("chat", "agentic"):
            subs = [r for r in c_recs if r["surface"] == sid and r["scheme"] == scheme]
            judge_counts: Dict[str, Dict[str, int]] = {}
            primary_vals = []
            for r in subs:
                labels = labels_of(r)
                det = _deterministic(surf, (r.get("response") or {}).get("text") or "", r)
                pv = _primary_value(surf, r, labels, det)
                if pv is not None:
                    primary_vals.append(pv)
                if labels:
                    for fld in judge_fields:
                        v = labels.get(fld)
                        if v is None:
                            continue
                        judge_counts.setdefault(fld, Counter())[str(v)] += 1
            baseline.append({
                "surface": sid, "scheme": scheme, "n": len(subs),
                "s_gen_mean": _mean([_probe_val(r, "s_gen") for r in subs]),
                "primary_mean": _mean(primary_vals),
                "judge_counts": {k: dict(v) for k, v in judge_counts.items()},
            })

    # ---- S3 position rates --------------------------------------------------
    def pos_rates(recs: List[Dict[str, Any]]) -> List[float]:
        counter = Counter()
        n_valid = 0
        for r in recs:
            order = list((r.get("variant") or {}).get("order") or list(range(len(headlines))))
            x = extract_picks((r.get("response") or {}).get("text") or "", headlines, order)
            if not x["parse_ok"]:
                continue
            n_valid += 1
            for pos in x["picked_positions"]:
                counter[pos] += 1
        return [counter[p] / n_valid if n_valid else 0.0 for p in range(1, 13)], n_valid

    s3_recs = [r for r in a_recs if r["surface"] == "s3_digest"]
    fwd_rates, n_fwd = pos_rates([r for r in s3_recs if r.get("order_arm") == "fwd"])
    rev_rates, n_rev = pos_rates([r for r in s3_recs if r.get("order_arm") == "rev"])
    bal_rates, n_bal = pos_rates(s3_recs)
    s3_position_rates = {"fwd": fwd_rates, "rev": rev_rates, "balanced": bal_rates}

    # S3 parse_ok rate
    def parse_ok_counts(recs: List[Dict[str, Any]]) -> Dict[str, int]:
        out = Counter()
        for r in recs:
            order = list((r.get("variant") or {}).get("order") or list(range(len(headlines))))
            x = extract_picks((r.get("response") or {}).get("text") or "", headlines, order)
            out["n"] += 1
            out["ok"] += int(x["parse_ok"])
        return dict(out)

    s3_parse = parse_ok_counts(s3_recs)

    # ---- S1 prefill compression ---------------------------------------------
    def bucket_span(recs: List[Dict[str, Any]]) -> Optional[float]:
        means = {}
        for bucket in BUCKETS:
            subs = [r for r in recs if r.get("bucket") == bucket]
            m = _mean([_probe_val(r, "s_gen") for r in subs])
            if m is not None:
                means[bucket] = m
        if len(means) == 3:
            return means["high"] - means["low"]
        return None

    s1_off = [r for r in a_recs if r["surface"] == "s1_speech" and r["scheme"] == "chat"]
    s1_on = [r for r in p_recs if r["surface"] == "s1_speech" and r["scheme"] == "chat"]
    chat_off_span = bucket_span(s1_off)
    chat_on_span = bucket_span(s1_on)
    ratio = (chat_on_span / chat_off_span) if (chat_off_span and chat_on_span) else None
    s1_prefill_compression = {
        "chat_off_span": chat_off_span,
        "chat_on_span": chat_on_span,
        "ratio": ratio,
        "n_items": len(FIXED_ITEMS),
    }

    # ---- calibration --------------------------------------------------------
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
        "s3_position_rates": s3_position_rates,
        "s3_parse_ok": s3_parse,
        "s3_position_rates_n": {"fwd": n_fwd, "rev": n_rev, "balanced": n_bal},
        "s1_prefill_compression": s1_prefill_compression,
        "calibration": calibration,
        "lean_ordinal_map": LEAN_ORDINAL_MAP,
        "prefill_strings": PREFILLS,
        "s_pre_invariant_violations": pre_violations,
    }

    # ---- images manifest + copy --------------------------------------------
    os.makedirs(IMAGES_DIR, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    items = load_items()
    for row in items:
        item = Item.from_dict(row)
        bucket = row["bucket"]
        cov = row["covariates"]
        coco_ids = cov.get("coco_ids") or []
        cats = cov.get("categories") or []
        for i, rel in enumerate(item.image_paths, start=1):
            src = os.path.join(ROOT, rel)
            fname = f"{bucket}_{item.item_id}_{i}.jpg"
            dst = os.path.join(IMAGES_DIR, fname)
            if not os.path.exists(dst) and os.path.exists(src):
                shutil.copy2(src, dst)
            manifest.append({
                "file": f"images/{fname}",
                "item_id": item.item_id,
                "bucket": bucket,
                "img_index": i,
                "coco_id": coco_ids[i - 1] if i - 1 < len(coco_ids) else None,
                "image_mean": item.image_scores[i - 1] if i - 1 < len(item.image_scores) else None,
                "categories": cats,
            })

    os.makedirs(ALL6_DIR, exist_ok=True)
    with open(ALL6_PATH, "w", encoding="utf-8") as handle:
        json.dump(_clean(all6), handle, ensure_ascii=False, indent=2)
    with open(ALL6_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(_clean(summary), handle, ensure_ascii=False, indent=2)
    with open(IMAGES_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(_clean(manifest), handle, ensure_ascii=False, indent=2)

    # ---- RESULTS.md ---------------------------------------------------------
    lines: List[str] = []
    lines.append("# round-9 six-surface pilot\n")

    lines.append("## 1. per-surface summary (arm A, image_mean correlations, n=18)\n")
    lines.append("| surface | scheme | image_mean->s_gen | image_mean->primary | image_mean->lean | refusal |")
    lines.append("|---|---|---|---|---|---|")
    for p in pearson:
        sid = p["surface"]
        scheme = p["scheme"]
        subs = [r for r in a_recs if r["surface"] == sid and r["scheme"] == scheme]
        ref = sum(1 for r in subs if r.get("refusal"))
        lines.append(f"| {sid} | {scheme} | {_f(p['image_mean_vs_s_gen'])} "
                     f"| {_f(p['image_mean_vs_primary'])} | {_f(p['image_mean_vs_lean_ordinal'])} "
                     f"| {ref}/{len(subs)} |")
    lines.append("")

    lines.append("## 2. judge lean / primary distribution by surface\n")
    for sid in SURFACES:
        surf = surfaces[sid]
        if surf.judge_spec is None:
            continue
        judge_fields = surf.judge_spec.fields
        agg = Counter()
        for r in a_recs:
            if r["surface"] != sid:
                continue
            labels = labels_of(r)
            if labels:
                for fld in judge_fields:
                    v = labels.get(fld)
                    if v is not None:
                        agg[fld][str(v)] += 1
        parts = []
        for fld in judge_fields:
            s = " ".join(f"{k}:{v}" for k, v in sorted(agg[fld].items(), key=lambda x: -x[1]))
            parts.append(f"{fld} [{s}]")
        lines.append(f"- {sid}: " + "  ".join(parts))
    lines.append("")

    lines.append("## 3. s_pre invariant (scheme x 18 items)\n")
    if pre_violations:
        lines.append(f"- **VIOLATIONS ({len(pre_violations)})**:")
        for v in pre_violations[:20]:
            lines.append(f"  - {v}")
    else:
        lines.append("- all six surfaces byte-identical within scheme (no violations)")
    lines.append("")

    lines.append("## 4. S3 position rates + parse_ok\n")
    lines.append("| pos | fwd | rev | balanced |")
    lines.append("|---|---|---|---|")
    for i in range(12):
        lines.append(f"| {i + 1} | {fwd_rates[i]:.3f} | {rev_rates[i]:.3f} | {bal_rates[i]:.3f} |")
    lines.append(f"- parse_ok: {s3_parse.get('ok', 0)}/{s3_parse.get('n', 0)} "
                 f"(fwd n={n_fwd}, rev n={n_rev}, balanced n={n_bal})")
    lines.append("")

    lines.append("## 5. S1 prefill compression (same 18 items)\n")
    lines.append(f"- chat off span: {_f(chat_off_span)}  chat on span: {_f(chat_on_span)}  "
                 f"ratio: {ratio if ratio is None else f'{ratio:.3f}'}")
    lines.append("")

    lines.append("## 6. refusal by (surface x scheme)\n")
    lines.append("| surface | scheme | arm A | arm C baseline |")
    lines.append("|---|---|---|---|")
    for sid in SURFACES:
        for scheme in ("chat", "agentic"):
            a = [r for r in a_recs if r["surface"] == sid and r["scheme"] == scheme]
            c = [r for r in c_recs if r["surface"] == sid and r["scheme"] == scheme]
            lines.append(f"| {sid} | {scheme} | {sum(1 for r in a if r.get('refusal'))}/{len(a)} "
                         f"| {sum(1 for r in c if r.get('refusal'))}/{len(c)} |")
    p_ref = sum(1 for r in p_recs if r.get("refusal"))
    lines.append(f"- arm P (s1 chat prefill on): {p_ref}/{len(p_recs)} refused")
    lines.append("")

    lines.append("## 7. delivery offset\n")
    if calibration.get("mad_over_sd") is not None:
        lines.append(f"- MAD / between-image sd = **{calibration['mad_over_sd']:.3f}** "
                     f"(MAD {calibration['mad']:.4f}, sd {calibration['sd_between_images']:.4f})")
    else:
        lines.append("- (no calibration.json)")
    lines.append("")

    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[analyze] wrote {ALL6_PATH}, {ALL6_SUMMARY_PATH}, {IMAGES_MANIFEST_PATH}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="run", help="calibrate,plan,run,analyze")
    parser.add_argument("--chunk", type=int, default=0, help="run chunk 0..3")
    parser.add_argument("--n-images", type=int, default=20)
    args = parser.parse_args()
    for phase in [p.strip() for p in args.phase.split(",") if p.strip()]:
        print(f"\n########## phase: {phase} ##########", flush=True)
        if phase == "calibrate":
            phase_calibrate(args.n_images)
        elif phase == "plan":
            phase_plan()
        elif phase == "run":
            phase_run(args.chunk)
        elif phase == "analyze":
            phase_analyze()
        else:
            raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    main()
