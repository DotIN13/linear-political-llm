"""Run the five forced-choice surfaces against the persona photos.

This replaces the hand-assembled prompts in ``probe_forced_choice.py``. That module
built its questions by reading the task packages' prompt files directly, which its own
docstring called a draft -- "if the pilot shows something, these become real
surfaces." They are real surfaces now, so this drives them through the registry and
every record is stamped with a ``measurement_rev`` that covers the prompt material.

  s9_neighborhood   pick 3 of 10          mean right_c of the picks
  s12_explain       pick 3 of 8, x2       code_pick_rel, on a live and an inert topic
  s11_health        rank all 6, x2        rank of the clinician route
  s10_groceries     call 1 of 8 tools     right_c of the platform called
  s14_outfits       pick 3 of 8, ranked   rank-weighted right_c

Seven question-units, 18 personas, two schemes, two orders each: **504 generations and
no judge calls**, because every one of these is read by rule.

THE ORDER CONTROL -- the one thing worth keeping from the old pilot
-------------------------------------------------------------------
Round 9 seeded presentation order on ``item_id``, so each bucket drew its own set of
orders. Position bias in the s3 task ran from 0.889 at position 1 to 0.139 at position
10; letting a seed decide which bucket gets which order pushes a bias that large
straight into the between-bucket contrast as variance.

Reversal does not fix it. Pairing an order with its exact reverse cancels the
anti-symmetric part -- the "earlier is better" trend -- and leaves the symmetric part
untouched, so a U-shaped bias with both primacy and recency survives completely.

So do not remove position bias, make it **identical across the buckets being
compared**. Every bucket gets the same rotations of the same base order, {0..5} x
{fwd, rev}. Position bias then lands as one constant offset on every bucket mean and
cancels exactly in any bucket difference.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench.paths import items_dir, runs_dir
from bench.registry import get_surface
from bench.store import measurement_rev
from bench.surfaces.registry import register_all
from bench.types import Item

register_all()

SEED = 20260908
BUCKETS = ("low", "mid", "high")
ITEMS_PER_BUCKET = 6
SCHEMES = ("chat", "agentic")
SURFACES = ["s9_neighborhood", "s12_explain", "s11_health", "s10_groceries", "s14_outfits"]

ITEMS_FILE = os.path.join(items_dir(), "explore_bucket_v1.jsonl")
OUT_DIR = runs_dir("probe_choice_five")


def trials_path() -> str:
    """One shard per job. See the note in ``bench/paths.py``: ``runs/`` is shared, and
    POSIX only guarantees an atomic append below 4096 bytes -- 17% of our real records
    are over that, so two jobs on one file would interleave mid-line."""
    job = os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
    return os.path.join(OUT_DIR, f"trials.{job}.jsonl")


def all_shards() -> List[str]:
    return sorted(_glob.glob(os.path.join(OUT_DIR, "trials.*.jsonl")))


# --------------------------------------------------------------------------- #
def load_items() -> List[Dict[str, Any]]:
    """Six personas per bucket, in file order. Deterministic, no sampling."""
    per: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(ITEMS_FILE, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "explore" and row.get("bucket") in BUCKETS:
                per[row["bucket"]].append(row)
    out: List[Dict[str, Any]] = []
    for bucket in BUCKETS:
        out.extend(per[bucket][:ITEMS_PER_BUCKET])
    return out


def matched_orders(n_options: int, n_personas: int = ITEMS_PER_BUCKET) -> List[Tuple[List[int], str]]:
    """The same (rotation, direction) set for every bucket. See the module docstring."""
    out: List[Tuple[List[int], str]] = []
    for r in range(n_personas):
        base = [(r + i) % n_options for i in range(n_options)]
        out.append((base, "fwd"))
        out.append((list(reversed(base)), "rev"))
    return out


def pool_size(surface: Any, qid: str) -> int:
    """How many options this surface shows for this question."""
    if hasattr(surface, "rows"):
        return len(surface.rows)
    if hasattr(surface, "by_topic"):
        return len(surface.by_topic[qid])
    return len(surface.by_scenario[qid])


def build_plan(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for sid in SURFACES:
        surface = get_surface(sid)()
        for qid in surface.question_ids():
            orders = matched_orders(pool_size(surface, qid))
            for scheme in SCHEMES:
                for row in items:
                    item = Item.from_dict(row)
                    peers = [r["item_id"] for r in items if r["bucket"] == row["bucket"]]
                    idx = peers.index(item.item_id)
                    for order, arm in (orders[idx * 2], orders[idx * 2 + 1]):
                        trial = surface.build(item, "photos", {
                            "scheme": scheme, "question": qid, "order": list(order)})
                        plan.append({
                            "surface": sid, "qid": qid, "scheme": scheme, "order_arm": arm,
                            "bucket": row["bucket"], "item_id": item.item_id,
                            "image_mean": item.image_mean, "trial": trial,
                            "_surface": surface,
                        })
    return plan


# --------------------------------------------------------------------------- #
def phase_run(limit: int = 0) -> int:
    from bench.adaptors.vllm_server import VLLMServerAdaptor

    os.makedirs(OUT_DIR, exist_ok=True)
    adaptor = VLLMServerAdaptor(
        model=os.environ.get("VLLM_SERVED_MODEL", "qwen3-vl-8b-instruct"),
        base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8073/v1"),
        seed=SEED)
    adaptor.setup()

    rev = measurement_rev(note=f"adaptor={adaptor.name}")
    print(f"[run] measurement_rev={rev}", flush=True)

    plan = build_plan(load_items())
    if limit:
        plan = plan[:limit]
    print(f"[run] {len(plan)} trials -> {trials_path()}", flush=True)

    n_err = n_unparsed = 0
    out_path = trials_path()
    with open(out_path, "a", encoding="utf-8") as handle:
        for i, entry in enumerate(plan):
            started = time.time()
            trial = entry["trial"]
            try:
                resp = adaptor.run(trial)
                # `run` RETURNS Response(error=...) rather than raising -- a payload
                # that will not build, an HTTP error, a timeout. The first version of
                # this loop set err=None unconditionally and threw that away, so 504
                # payload-build failures were written to disk as clean empty records
                # with error=None. Read the field.
                text_out, err = resp.text, resp.error
                if err:
                    n_err += 1
                    resp = None
            except Exception as exc:                        # noqa: BLE001
                text_out, err, resp = "", f"{type(exc).__name__}: {exc}", None
                n_err += 1

            read: Dict[str, Any] = {}
            if resp is not None:
                outcome = entry["_surface"].extract(resp, trial)
                read = dict(getattr(outcome, "extra", {}) or {})
            if not read.get("parsed"):
                n_unparsed += 1

            rec = {
                "surface": entry["surface"], "qid": entry["qid"], "scheme": entry["scheme"],
                "order_arm": entry["order_arm"], "bucket": entry["bucket"],
                "item_id": entry["item_id"], "image_mean": entry["image_mean"],
                "order": list(trial.variant.get("order") or []),
                "text": text_out, "error": err, "ms": (time.time() - started) * 1000.0,
                "measurement_rev": rev, "adaptor": adaptor.name, "model": adaptor.model,
                "seed": SEED,
                "trial_key": f"{entry['surface']}/{entry['qid']}/{entry['scheme']}/"
                             f"{entry['order_arm']}/{entry['item_id']}/{rev}",
                **read,
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(plan):
                print(f"[run] {i + 1}/{len(plan)}  errors={n_err}  unparsed={n_unparsed}",
                      flush=True)

    print(f"[run] done. errors={n_err} unparsed={n_unparsed} -> {out_path}", flush=True)
    if plan and n_err == len(plan):
        print("[run] EVERY trial errored -- this is a broken run, not a result.",
              file=sys.stderr, flush=True)
        return 2
    if plan and n_unparsed == len(plan):
        print("[run] NOTHING parsed. Check the first record's text before believing "
              "any of this.", file=sys.stderr, flush=True)
        return 2
    return 1 if n_err else 0


def phase_plan() -> int:
    """Build every trial and print the shape. No GPU, no model."""
    plan = build_plan(load_items())
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for e in plan:
        counts[(e["surface"], e["qid"])] += 1
    print(f"{len(plan)} trials")
    for (sid, qid), n in sorted(counts.items()):
        print(f"  {sid:20} {qid:12} {n:5}")
    return 0


def phase_report() -> int:
    rows: List[Dict[str, Any]] = []
    for shard in all_shards():
        rows += [json.loads(l) for l in open(shard, encoding="utf-8") if l.strip()]
    if not rows:
        print(f"no records under {OUT_DIR}", file=sys.stderr)
        return 1
    print(json.dumps({"n": len(rows), "shards": len(all_shards())}, indent=2))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", default="plan", help="plan | run | report")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    codes = []
    for phase in a.phase.split(","):
        phase = phase.strip()
        if phase == "plan":
            codes.append(phase_plan())
        elif phase == "run":
            codes.append(phase_run(a.limit))
        elif phase == "report":
            codes.append(phase_report())
        else:
            raise SystemExit(f"unknown phase {phase!r}")
    raise SystemExit(max(codes) if codes else 0)


if __name__ == "__main__":
    main()
