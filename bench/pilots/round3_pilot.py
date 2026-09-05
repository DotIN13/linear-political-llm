"""Round-3 pilot driver (docs/bench/09).

Three phases:

* ``items``     -- build the 12 pilot items (4 per stratum: 0 low, 5 mid, 9 high).
* ``calibrate`` -- delivery-offset calibration: 20 images x 2 schemes, one forward
  each, reporting MAD / between-image sd (board-tasks: this offset is a factor,
  never a condition effect).
* ``analyze``   -- read ``runs/<run>/trials.jsonl`` + ``judges.jsonl`` and print
  the round's only real numbers: ``stratum -> s_pre`` / ``stratum -> s_gen`` per
  surface x scheme, the with/without-image baseline difference, and the judge
  field-level distributions (with the board's degeneracy test).

Nothing here touches the GPU except ``calibrate`` and the generation run itself
(the latter is ``bench run``, driven by the sbatch). The judge + analyze phases
are CPU/API only and run after the GPU job.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bench.adaptors.local_hf import token_scoring, split_probe_id  # noqa: E402
from bench.surfaces.generation import (  # noqa: E402
    AGENTIC_ACK, AGENTIC_OPENER, FILENAMES, FILENAMES_LINE, SYSTEM_AGENTIC, TOOLS,
)

OUT_DIR = os.path.join(ROOT_DIR, "runs", "pilot_round3")
PILOT_ITEMS = os.path.join(ROOT_DIR, "items", "pilot_round3.jsonl")
EXPLORE = os.path.join(ROOT_DIR, "items", "explore_v2.jsonl")

PROBE_ID = "combined_ideology_headwise_linear"
MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
MODEL_FAMILY = "qwen3-vl"
DEFAULT_K = 16

PILOT_STRATA = (0, 5, 9)
ITEMS_PER_STRATUM = 4

NEUTRAL_QUESTION = "What stands out to you in these photos?"


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


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #
def phase_items(seed: int = 42) -> None:
    import random
    by_stratum: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with open(EXPLORE, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_stratum[row["stratum"]].append(row)
    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    for stratum in PILOT_STRATA:
        pool = list(by_stratum.get(stratum, []))
        rng.shuffle(pool)
        picked.extend(pool[:ITEMS_PER_STRATUM])
    os.makedirs(os.path.dirname(PILOT_ITEMS), exist_ok=True)
    with open(PILOT_ITEMS, "w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[items] wrote {len(picked)} items -> {PILOT_ITEMS}")
    for s in PILOT_STRATA:
        n = sum(1 for r in picked if r["stratum"] == s)
        print(f"  stratum {s}: {n} items")


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def phase_calibrate(n_images: int = 20) -> None:
    import numpy as np
    import torch

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

    # collect n_images image paths spanning the whole stratum range (2 per
    # stratum, round-robin) so the between-image sd reflects the real spread,
    # not a single left-leaning stratum.
    per_stratum: Dict[int, List[str]] = defaultdict(list)
    with open(EXPLORE, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            per_stratum[row["stratum"]].extend(row["image_paths"])
    image_paths: List[str] = []
    strata = sorted(per_stratum)
    i = 0
    while len(image_paths) < n_images and i < max(len(v) for v in per_stratum.values()):
        for s in strata:
            if i < len(per_stratum[s]) and len(image_paths) < n_images:
                image_paths.append(per_stratum[s][i])
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
# analyze
# --------------------------------------------------------------------------- #
def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = shared
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else None


def _read_trials(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "trials.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_judges(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "judges.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def phase_analyze(run_dir: str) -> None:
    trials = _read_trials(run_dir)
    judges = _read_judges(run_dir)
    if not trials:
        print(f"no trials in {run_dir}", file=sys.stderr)
        raise SystemExit(1)

    judge_by_key = {j["trial_key"]: j for j in judges}

    lines: List[str] = []
    lines.append("# round-3 pilot analysis\n")

    # --- calibration ---
    calib_path = os.path.join(run_dir, "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path, encoding="utf-8") as handle:
            calib = json.load(handle)
        lines.append(f"## delivery offset (MAD / between-image sd, k={calib['k']})")
        lines.append(f"- MAD / sd = **{calib['mad_over_sd']:.3f}**  "
                     f"(mean offset {calib['mean_offset']:+.4f}, MAD {calib['mad']:.4f}, "
                     f"sd {calib['sd_between_images']:.4f}, n={calib['n_images']})")
        lines.append("")

    # --- stratum -> s_pre / s_gen ---
    gen = [r for r in trials if r.get("probe") and r["probe"].get("s_pre") is not None
           and r.get("condition") == "C"]
    base = [r for r in trials if r.get("probe") and r["probe"].get("s_pre") is not None
            and r.get("condition") == "E"]

    lines.append("## stratum -> s_pre / s_gen (spearman, condition C)\n")
    lines.append("| surface | scheme | rho(stratum, s_pre) | rho(stratum, s_gen) | n |")
    lines.append("|---|---|---|---|---|")
    by_cell: Dict[str, Dict[str, Any]] = {}
    for r in gen:
        cell = (r["surface"], r["variant"].get("scheme"))
        by_cell.setdefault(cell, {"strata": [], "s_pre": [], "s_gen": []})
        by_cell[cell]["strata"].append(r["stratum"])
        by_cell[cell]["s_pre"].append(r["probe"]["s_pre"])
        by_cell[cell]["s_gen"].append(r["probe"]["s_gen"])
    for cell in sorted(by_cell, key=str):
        d = by_cell[cell]
        rho_pre = _spearman(d["strata"], d["s_pre"])
        rho_gen = _spearman(d["strata"], d["s_gen"])
        surface, scheme = cell
        lines.append(f"| {surface} | {scheme} | {_f(rho_pre)} | {_f(rho_gen)} | {len(d['strata'])} |")
    lines.append("")

    # --- with/without image baseline difference ---
    lines.append("## with-image vs no-image baseline (condition C vs E, mean)\n")
    lines.append("| surface | scheme | C s_pre | E s_pre | C s_gen | E s_gen |")
    lines.append("|---|---|---|---|---|---|")
    def cell_mean(rows, field):
        vals = [r["probe"][field] for r in rows if r["probe"].get(field) is not None]
        return sum(vals) / len(vals) if vals else None
    for cell in sorted({(r["surface"], r["variant"].get("scheme")) for r in trials}, key=str):
        surface, scheme = cell
        c = [r for r in gen if r["surface"] == surface and r["variant"].get("scheme") == scheme]
        e = [r for r in base if r["surface"] == surface and r["variant"].get("scheme") == scheme]
        lines.append(f"| {surface} | {scheme} | {_f(cell_mean(c, 's_pre'))} | {_f(cell_mean(e, 's_pre'))} "
                     f"| {_f(cell_mean(c, 's_gen'))} | {_f(cell_mean(e, 's_gen'))} |")
    lines.append("")

    # --- judge distributions ---
    if judge_by_key:
        lines.append("## judge field distributions (degeneracy: one bucket >60% or <=3 buckets used)\n")
        from bench.judges import judge_specs
        specs = judge_specs()
        for surface in sorted(specs):
            subs = [j for j in judges if j.get("surface") == surface and "labels" in j]
            if not subs:
                continue
            spec = specs[surface]
            lines.append(f"### {surface}  (n={len(subs)}, judge={spec.model})")
            label_fields = [f for f in spec.fields if f not in ("named_attributes",)]
            for field in label_fields:
                if field == "topic_slug":
                    continue
                vals = [j["labels"].get(field) for j in subs if j.get("labels", {}).get(field) is not None]
                if not vals:
                    continue
                dist: Dict[str, int] = defaultdict(int)
                for v in vals:
                    dist[str(v)] += 1
                total = len(vals)
                top = max(dist.values())
                n_buckets = len(dist)
                degenerate = top / total > 0.6 or n_buckets <= 3
                rendered = "  ".join(f"{k}:{dist[k]}" for k in sorted(dist, key=lambda x: -dist[x]))
                flag = " **DEGENERATE**" if degenerate else ""
                lines.append(f"- `{field}` ({n_buckets} buckets, max {top}/{total}={top/total:.2f}){flag}: {rendered}")
            lines.append("")
    else:
        lines.append("## judge\n(no judges.jsonl found)\n")

    report_path = os.path.join(run_dir, "RESULTS.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[analyze] wrote {report_path}")


def _f(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", default="items", help="items,calibrate,analyze")
    parser.add_argument("--run", default=OUT_DIR, help="run dir for analyze")
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
        elif phase == "analyze":
            phase_analyze(args.run)
        else:
            raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    main()
