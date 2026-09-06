"""S3 pilot (docs/bench/13): make the digest surface real and take a first reading.

Arms (all attribution="shown", R1-style prefill to kill refusals):

* ``C``  -- with images, both schemes, the 9 round-4 items (3/bucket):
    * chat    x 9 items x 2 reps (rep 1 is the within-session stability repeat)
    * agentic x 9 items x 1 rep
* ``E``  -- no-image baseline, both schemes x 4 seeds.

Total 35 generations, 600-token cap. ``run`` is the GPU phase; ``analyze`` is
CPU and re-extracts the picks from the stored text, so the extractor (the part
most likely to need tuning) can be iterated without re-running the GPU.

Prefill tokens are appended after the generation prompt (so they are *input*, not
generated, and are therefore excluded from ``s_gen``), and ``mm_token_type_ids``
is padded with zeros to match (docs/bench/12 section 7 pitfalls 1 and 3).
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
from bench.adaptors.local_hf import LocalHFAdaptor, token_scoring  # noqa: E402
from bench.store import git_rev, measurement_rev  # noqa: E402
from bench.surfaces.generation import (  # noqa: E402
    S3_MATCH_THRESHOLD, S3_AMBIGUITY_MARGIN, detect_refusal, _refusal_match,
    extract_picks, load_s3_headlines, word_count,
)
from bench.types import BASELINE_ITEM_ID  # noqa: E402

MODEL_PATH = "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
MODEL_FAMILY = "qwen3-vl"
PROBE_ID = "combined_ideology_headwise_linear"
TOP_K = 16
MAX_NEW_TOKENS = 600
PREFILL = "Here are the five I'd show you:\n\n"
ATTRIBUTION = "shown"

OUT_DIR = os.path.join(ROOT, "runs", "pilot_s3")
ITEMS_FILE = os.path.join(ROOT, "items", "pilot_round4.jsonl")
TRIALS_PATH = os.path.join(OUT_DIR, "trials.jsonl")
RESULTS_PATH = os.path.join(OUT_DIR, "RESULTS.md")

BUCKETS = ("low", "mid", "high")
BASELINE_SEEDS = (0, 1, 2, 3)
C_SEED = 42


def load_items() -> Dict[str, Dict[str, Any]]:
    rows = {}
    with open(ITEMS_FILE, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["item_id"]] = r
    return rows


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run_generation(adaptor, trial, prefill: str, seed: int):
    """Generate + read s_pre/s_gen/s_img with the R1 prefill (docs/bench/12 S7)."""
    import numpy as np
    import torch

    ts = token_scoring()
    started = time.time()

    meta = trial.meta or {}
    tools = meta.get("tools")
    prefix_n = int(meta.get("prefix_n_messages", len(trial.conversation.messages) - 1))

    messages = ts.resolve_messages_images(
        trial.conversation.messages, image_root=ROOT, cache_dir=adaptor.resized_cache_dir
    )
    if messages is None:
        return {"error": "image preparation failed", "timing_ms": (time.time() - started) * 1000.0}

    prefix = messages[:prefix_n]
    full = adaptor._encode(messages, tools=tools, add_generation_prompt=True)
    pref = adaptor._encode(prefix, tools=tools, add_generation_prompt=False)
    k = int(pref["input_ids"][0].numel()) - 1

    if prefill:
        pids = adaptor.processor.tokenizer.encode(prefill, add_special_tokens=False)
        full["input_ids"] = torch.cat(
            [full["input_ids"], torch.tensor([pids], dtype=full["input_ids"].dtype)], dim=1
        )
        if "attention_mask" in full:
            full["attention_mask"] = torch.cat(
                [full["attention_mask"],
                 torch.ones((1, len(pids)), dtype=full["attention_mask"].dtype)], dim=1
            )
        if "mm_token_type_ids" in full:
            full["mm_token_type_ids"] = torch.cat(
                [full["mm_token_type_ids"],
                 torch.zeros((1, len(pids)), dtype=full["mm_token_type_ids"].dtype)], dim=1
            )

    runtime = ts.build_probe_runtime(
        model=adaptor.hf_model, probe=adaptor.probe, top_k=TOP_K,
        mode=adaptor.mode, model_family=adaptor.model_family,
    )
    union = sorted(runtime["module_names"])
    named_modules = dict(adaptor.hf_model.named_modules())
    missing = [n for n in union if n not in named_modules]
    if missing:
        return {"error": f"missing probe modules: {missing[:3]}",
                "timing_ms": (time.time() - started) * 1000.0}

    logs = {n: [] for n in union}

    def make_hook(name):
        def hook_fn(_m, _i, out):
            tensor = out[0] if isinstance(out, tuple) else out
            logs[name].append(tensor.detach().to(dtype=torch.float32).cpu())
        return hook_fn

    hooks = [named_modules[n].register_forward_hook(make_hook(n)) for n in union]
    try:
        with torch.no_grad():
            out = adaptor.hf_model.generate(
                **ts.move_to_device(full, adaptor.hf_model),
                max_new_tokens=trial.max_new_tokens,
                do_sample=False,
            )
    finally:
        for hook in hooks:
            hook.remove()

    input_ids = full["input_ids"][0].cpu().numpy()
    prefill_len = int(len(input_ids))
    generated = out[0][prefill_len:]
    text = adaptor.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    image_mask = np.isin(input_ids, np.asarray(sorted(adaptor.image_token_ids), dtype=np.int64)) \
        if adaptor.image_token_ids else np.zeros_like(input_ids, dtype=bool)
    n_image_tokens = int(image_mask.sum())

    def score(captured):
        return ts.score_from_captured(captured, runtime)[0]

    prefill_captured = {n: logs[n][0] for n in union}
    decode_captured = [{n: logs[n][j] for n in union} for j in range(1, len(logs[union[0]]))]

    prefill_primary = score(prefill_captured).cpu().numpy()
    s_pre = float(prefill_primary[k])
    s_img = float(prefill_primary[image_mask].mean()) if n_image_tokens else None

    gen_vals = [float(score(dc).cpu().numpy()[-1]) for dc in decode_captured]
    s_gen = float(np.asarray(gen_vals).mean()) if gen_vals else None

    return {
        "text": text,
        "s_pre": s_pre,
        "s_gen": s_gen,
        "s_img": s_img,
        "n_generated_tokens": int(len(generated)),
        "n_image_tokens": n_image_tokens,
        "timing_ms": (time.time() - started) * 1000.0,
        "top_k": TOP_K,
    }


def phase_run() -> None:
    registry.load_all()
    s3 = registry.get_surface("s3_digest")()
    items = load_items()

    plan: List[Dict[str, Any]] = []

    # Arm C: with images.
    for row in items.values():
        for scheme in ("chat", "agentic"):
            reps = (0, 1) if scheme == "chat" else (0,)
            for rep in reps:
                trial = s3.build(
                    __item(row), "C", {"scheme": scheme, "attribution": ATTRIBUTION}, seed=C_SEED
                )
                plan.append({
                    "arm": "C", "condition": "C", "scheme": scheme, "rep": rep,
                    "item_id": row["item_id"], "bucket": row["bucket"],
                    "image_scores": row["image_scores"],
                    "seed": C_SEED, "trial": trial,
                })

    # Arm E: no-image baseline, both schemes x 4 seeds.
    from bench.types import Item
    baseline = Item(item_id=BASELINE_ITEM_ID, images=[], image_paths=[], image_scores=[],
                    stratum=-1, covariates={}, split="explore")
    for scheme in ("chat", "agentic"):
        for seed in BASELINE_SEEDS:
            trial = s3.build(
                baseline, "E", {"scheme": scheme, "attribution": ATTRIBUTION}, seed=seed
            )
            plan.append({
                "arm": "E", "condition": "E", "scheme": scheme, "rep": 0,
                "item_id": BASELINE_ITEM_ID, "bucket": None,
                "image_scores": [], "seed": seed, "trial": trial,
            })

    n_by_arm = Counter(p["arm"] for p in plan)
    print(f"[run] {len(items)} items -> {len(plan)} trials "
          f"(C={n_by_arm['C']}, E={n_by_arm['E']})", flush=True)

    adaptor = LocalHFAdaptor(
        model=MODEL_PATH, probe=PROBE_ID, top_k=TOP_K, seed=C_SEED,
        data_dir=os.path.join(ROOT, "results", "probes"),
        resized_cache_dir=os.path.join(ROOT, "items", "_image_cache"),
    )
    adaptor.setup()
    import torch
    torch.manual_seed(C_SEED)

    probe_weights = adaptor.describe().get("probe_weights")
    rev = measurement_rev(ROOT, extra_files=[probe_weights] if probe_weights else [],
                          note=f"top_k={TOP_K}")
    code_rev = git_rev(ROOT)
    print(f"[run] code_rev={code_rev}  measurement_rev={rev}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(TRIALS_PATH):
        with open(TRIALS_PATH, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r["arm"], r["scheme"], r["item_id"], r["seed"], r["rep"]))

    n = 0
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        for entry in plan:
            key = (entry["arm"], entry["scheme"], entry["item_id"], entry["seed"], entry["rep"])
            if key in done:
                continue
            torch.manual_seed(entry["seed"])
            res = run_generation(adaptor, entry["trial"], PREFILL, entry["seed"])
            trial = entry["trial"]
            record = {
                "arm": entry["arm"],
                "condition": entry["condition"],
                "scheme": entry["scheme"],
                "rep": entry["rep"],
                "item_id": entry["item_id"],
                "bucket": entry["bucket"],
                "image_scores": entry["image_scores"],
                "seed": entry["seed"],
                "variant": trial.variant,
                "attribution": ATTRIBUTION,
                "text": res.get("text"),
                "error": res.get("error"),
                "refusal": detect_refusal(res.get("text") or ""),
                "refusal_match": _refusal_match(res.get("text") or ""),
                "word_count": word_count(res.get("text") or ""),
                "probe": {kk: res.get(kk) for kk in
                          ("s_pre", "s_gen", "s_img", "n_generated_tokens",
                           "n_image_tokens", "timing_ms", "top_k")},
                "measurement_rev": rev,
                "code_rev": code_rev,
                "model": adaptor.model,
                "prefill": PREFILL,
                "s3_match_threshold": S3_MATCH_THRESHOLD,
                "s3_ambiguity_margin": S3_AMBIGUITY_MARGIN,
            }
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}] {entry['arm']}/{entry['scheme']}/{entry['item_id']} "
                  f"rep={entry['rep']} refusal={record['refusal']} "
                  f"s_gen={res.get('s_gen') if res.get('s_gen') is None else round(res['s_gen'], 4)} "
                  f"tokens={res.get('n_generated_tokens')}", flush=True)

    adaptor.teardown()
    print(f"[run] wrote {n} new records -> {TRIALS_PATH}", flush=True)


def __item(row: Dict[str, Any]):
    from bench.types import Item
    return Item.from_dict(row)


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def _read_trials() -> List[Dict[str, Any]]:
    if not os.path.exists(TRIALS_PATH):
        return []
    rows = []
    with open(TRIALS_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


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


def _extract(record: Dict[str, Any], headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
    order = list(record["variant"].get("order") or list(range(len(headlines))))
    return extract_picks(record.get("text") or "", headlines, order)


def phase_analyze() -> None:
    registry.load_all()
    headlines = load_s3_headlines()
    trials = _read_trials()
    if not trials:
        print("no trials; run the GPU phase first", file=sys.stderr)
        raise SystemExit(1)

    c_recs = [r for r in trials if r["arm"] == "C"]
    e_recs = [r for r in trials if r["arm"] == "E"]

    lines: List[str] = []
    lines.append("# S3 pilot (docs/bench/13)\n")

    # --- 1. refusal rate by scheme x bucket (arm C) + E ----------------------
    lines.append("## 1. refusal rate\n")
    lines.append("| scheme | low | mid | high | C total |")
    lines.append("|---|---|---|---|---|")
    for scheme in ("chat", "agentic"):
        by_bucket = {}
        for bucket in BUCKETS:
            subs = [r for r in c_recs if r["scheme"] == scheme and r["bucket"] == bucket
                    and r["rep"] == 0]
            n_ref = sum(1 for r in subs if r["refusal"])
            by_bucket[bucket] = f"{n_ref}/{len(subs)}"
        total = [r for r in c_recs if r["scheme"] == scheme and r["rep"] == 0]
        n_ref = sum(1 for r in total if r["refusal"])
        lines.append(f"| {scheme} | {by_bucket['low']} | {by_bucket['mid']} | "
                     f"{by_bucket['high']} | {n_ref}/{len(total)} |")
    for scheme in ("chat", "agentic"):
        subs = [r for r in e_recs if r["scheme"] == scheme]
        n_ref = sum(1 for r in subs if r["refusal"])
        lines.append(f"- E/{scheme}: {n_ref}/{len(subs)} refused")
    lines.append("")

    # --- 2. parse failure rate + detail --------------------------------------
    lines.append("## 2. parse failures\n")
    for arm, recs in (("C", c_recs), ("E", e_recs)):
        # main records only (rep 0 for C)
        main = [r for r in recs if r["rep"] == 0]
        extracted = [(r, _extract(r, headlines)) for r in main]
        fails = [(r, x) for r, x in extracted if not x["parse_ok"]]
        lines.append(f"- {arm}: parse_ok=False {len(fails)}/{len(main)} "
                     f"({len(fails) / max(1, len(main)):.1%}); "
                     f"match_methods: {dict(Counter(x['match_method'] for _, x in extracted))}")
        for r, x in fails:
            text = (r.get("text") or "")[:200].replace("\n", " ")
            lines.append(f"  - {arm}/{r['scheme']}/{r['item_id']} rep={r['rep']} "
                         f"n_picked={x['n_picked']} method={x['match_method']} "
                         f"min_match={x['min_match_score'] if x['min_match_score'] is None else round(x['min_match_score'], 3)} "
                         f"hids={x['picked_hids']} | \"{text}\"")
    lines.append("")

    # --- 3. position preference ---------------------------------------------
    lines.append("## 3. position preference (1..12)\n")
    lines.append("| pos | n_picked | rate | rate - 5/12 |")
    lines.append("|---|---|---|---|")
    pos_counter = Counter()
    n_valid = 0
    for r in c_recs:
        if r["rep"] != 0:
            continue
        x = _extract(r, headlines)
        if not x["parse_ok"]:
            continue
        n_valid += 1
        for p in x["picked_positions"]:
            pos_counter[p] += 1
    for p in range(1, 13):
        n = pos_counter[p]
        rate = n / n_valid if n_valid else 0.0
        lines.append(f"| {p} | {n} | {rate:.3f} | {rate - 5/12:+.3f} |")
    lines.append(f"- n_valid (parse_ok) C records used: {n_valid}; expected rate 5/12 = {5/12:.3f}")
    lines.append("")

    # --- 4. stability: chat/C 9 items x 2 reps ------------------------------
    lines.append("## 4. stability (chat/C, rep 0 vs rep 1)\n")
    chat = [r for r in c_recs if r["scheme"] == "chat"]
    by_item = defaultdict(dict)
    for r in chat:
        by_item[r["item_id"]][r["rep"]] = r
    identical = 0
    both_ok = 0
    n_right_deltas: List[int] = []
    lines.append("| item | same_hids | n_right(0) | n_right(1) |")
    lines.append("|---|---|---|---|")
    for item_id in sorted(by_item):
        r0 = by_item[item_id].get(0)
        r1 = by_item[item_id].get(1)
        if r0 is None or r1 is None:
            continue
        x0 = _extract(r0, headlines)
        x1 = _extract(r1, headlines)
        if x0["parse_ok"] and x1["parse_ok"]:
            both_ok += 1
            same = sorted(x0["picked_hids"]) == sorted(x1["picked_hids"])
            identical += int(same)
            n_right_deltas.append(abs(int(x0["n_right"]) - int(x1["n_right"])))
            lines.append(f"| {item_id} | {same} | {x0['n_right']} | {x1['n_right']} |")
        else:
            lines.append(f"| {item_id} | parse_fail (ok0={x0['parse_ok']} ok1={x1['parse_ok']}) | - | - |")
    lines.append(f"- identical picked_hids: {identical}/{both_ok} ({identical / max(1, both_ok):.0%})")
    lines.append(f"- |Delta n_right| across items: {n_right_deltas} "
                 f"(sum={sum(n_right_deltas)}, mean={_mean(n_right_deltas):.2f})")
    lines.append("")

    # --- 5. DV1/DV2 by bucket x scheme + image_mean correlations -------------
    lines.append("## 5. DV1 (slant_c_mean) and DV2 (n_right)\n")
    lines.append("| scheme | bucket | DV1 mean | DV2 mean | n (parse_ok) |")
    lines.append("|---|---|---|---|---|")
    per = defaultdict(list)
    for scheme in ("chat", "agentic"):
        for bucket in BUCKETS:
            subs = [r for r in c_recs if r["scheme"] == scheme and r["bucket"] == bucket
                    and r["rep"] == 0]
            exs = [_extract(r, headlines) for r in subs]
            ok = [x for x in exs if x["parse_ok"]]
            dv1 = _mean([x["slant_c_mean"] for x in ok])
            dv2 = _mean([x["n_right"] for x in ok])
            lines.append(f"| {scheme} | {bucket} | {_f(dv1)} | {_f(dv2)} | {len(ok)} |")
            for x in ok:
                per[(scheme, bucket)].append(x)
    lines.append("")
    for scheme in ("chat", "agentic"):
        subs = [r for r in c_recs if r["scheme"] == scheme and r["rep"] == 0]
        pairs = [(r["image_scores"], _extract(r, headlines)) for r in subs]
        im = [sum(r[0]) / len(r[0]) for r in pairs if r[1]["parse_ok"]]
        dv1 = [r[1]["slant_c_mean"] for r in pairs if r[1]["parse_ok"]]
        dv2 = [r[1]["n_right"] for r in pairs if r[1]["parse_ok"]]
        lines.append(f"- {scheme}: image_mean->DV1 pearson={_f(_corr(im, dv1))} "
                     f"(n={len(im)}); image_mean->DV2 pearson={_f(_corr(im, dv2))}")
    lines.append("")

    # --- 6. dropped_topics distribution -------------------------------------
    lines.append("## 6. dropped_topics distribution\n")
    topic_counter = Counter()
    all_topics = {h["topic"] for h in headlines}
    for r in c_recs:
        if r["rep"] != 0:
            continue
        x = _extract(r, headlines)
        if not x["parse_ok"]:
            continue
        for t in x["dropped_topics"]:
            topic_counter[t] += 1
    lines.append("| topic | n_dropped (out of parse_ok C records) |")
    lines.append("|---|---|")
    for topic in sorted(all_topics):
        lines.append(f"| {topic} | {topic_counter.get(topic, 0)} |")
    lines.append(f"- all_topics={sorted(all_topics)}")
    lines.append("")

    # --- 7. probes -----------------------------------------------------------
    lines.append("## 7. probes (rep 0)\n")
    lines.append("| scheme | arm | bucket | s_pre | s_gen | s_img | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for scheme in ("chat", "agentic"):
        for bucket in BUCKETS:
            subs = [r for r in c_recs if r["scheme"] == scheme and r["bucket"] == bucket
                    and r["rep"] == 0]
            sp = _mean([r["probe"]["s_pre"] for r in subs if r["probe"]["s_pre"] is not None])
            sg = _mean([r["probe"]["s_gen"] for r in subs if r["probe"]["s_gen"] is not None])
            si = _mean([r["probe"]["s_img"] for r in subs if r["probe"]["s_img"] is not None])
            lines.append(f"| {scheme} | C | {bucket} | {_f(sp)} | {_f(sg)} | {_f(si)} | {len(subs)} |")
        esubs = [r for r in e_recs if r["scheme"] == scheme]
        sp = _mean([r["probe"]["s_pre"] for r in esubs if r["probe"]["s_pre"] is not None])
        sg = _mean([r["probe"]["s_gen"] for r in esubs if r["probe"]["s_gen"] is not None])
        lines.append(f"| {scheme} | E | - | {_f(sp)} | {_f(sg)} | - | {len(esubs)} |")
    lines.append("")

    # --- truncation / word count --------------------------------------------
    lines.append("## truncation + word count\n")
    for arm, recs in (("C", c_recs), ("E", e_recs)):
        main = [r for r in recs if r["rep"] == 0]
        wc = [r["word_count"] for r in main]
        trunc = sum(1 for r in main if r["probe"]["n_generated_tokens"] >= MAX_NEW_TOKENS)
        lines.append(f"- {arm} (n={len(main)}): truncated {trunc}/{len(main)}; "
                     f"word_count mean={_mean(wc):.0f} min={min(wc) if wc else '-'} "
                     f"max={max(wc) if wc else '-'}")
    lines.append("")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[analyze] wrote {RESULTS_PATH}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="run,analyze")
    args = parser.parse_args()
    for phase in [p.strip() for p in args.phase.split(",") if p.strip()]:
        print(f"\n########## phase: {phase} ##########", flush=True)
        if phase == "run":
            phase_run()
        elif phase == "analyze":
            phase_analyze()
        else:
            raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    main()
