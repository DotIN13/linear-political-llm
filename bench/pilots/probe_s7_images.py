"""Does showing ten photos instead of three change the s7 result?

One wording -- `b_committed`, which round 16 chose -- and one thing varied: how
many photos the persona is built from. **The wording is imported from
`bench.surfaces.groupchat`, never retyped here**, so this probe cannot drift away
from the surface it is testing.

    3 photos  x  4 questions  x  8 personas  =  32
    10 photos x  4 questions  x  8 personas  =  32
                                       total    64 generations

**Both arms run here.** The 3-photo arm is not taken from round 16, for three
reasons that each on their own would break the comparison: round 16 ran on an
H100 whose numbers are not on the same footing as a fresh server, `generation.py`
has changed since so the measurement fingerprint differs, and 32 answers is too
few for the difference between arms to survive a change of hardware as well.

**What this cannot separate, and it is not a small thing.** In the agentic
scheme every photo is one `view_image` turn, so ten photos is **27 turns against
three photos' 13**. Photo count and transcript length move together and there is
no version of this design in which they do not. So a difference between the arms
is "ten photos in a longer transcript" versus "three photos in a shorter one" --
not the photo count alone.

**And the manipulation is expected to get weaker, not stronger.** An item's
score is the mean of its photos' scores, so averaging ten draws from a bucket
sits closer to that bucket's centre than averaging three. The extreme tails
flatten. `plan` prints the achieved left/right gap for both arms **before any GPU
is spent**, because a narrower gap in the 10-photo arm would look exactly like
"more photos does not help" while actually being a weaker treatment.

Scratch, like every pilot: nothing here is under `bench/surfaces/`, so choosing
to keep or drop this changes no fingerprint.

    python -m bench.pilots.probe_s7_images plan
    python -m bench.pilots.probe_s7_images run
    python -m bench.pilots.probe_s7_images judge
    python -m bench.pilots.probe_s7_images report
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench import registry
from bench.surfaces.generation import build_scheme_messages
from bench.surfaces.groupchat import QUESTION_TEMPLATE
from bench.types import Conversation, Item, Trial

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "runs", "probe_s7_images")
LOG_PATH = os.path.join(OUT_DIR, "log.jsonl")
JUDGED_PATH = os.path.join(OUT_DIR, "judged.jsonl")
MESSAGES_FILE = os.path.join(ROOT, "bench", "data", "s7_family_chat_v1.json")

# One items file per arm. The 10-photo file does not exist until the sampler is
# run with --images-per-item 10; `plan` says so rather than failing obscurely.
ITEMS_FILES: Dict[int, str] = {
    3: os.path.join(ROOT, "items", "explore_bucket_v1.jsonl"),
    10: os.path.join(ROOT, "items", "explore_bucket_n10_v1.jsonl"),
}
ARMS: Tuple[int, ...] = (3, 10)

SEED = 20260907
N_PER_SIDE = 4
MAX_NEW_TOKENS = 320
SCHEME = "agentic"
# Must match the server's own --max-model-len; used only to warn in `plan`.
MAX_MODEL_LEN = 8192

# The same four as round 16, so the questions are not a new variable: two that
# collapsed in round 15 and were rewritten, one middling case, and one where
# agentic specifically collapsed while chat did not.
QUESTION_IDS = ["m02", "m09", "m01", "m08"]

JUDGE_ID = "s2_proposal"


def question_text(message: str) -> str:
    """The surface's own template, formatted. Never a local copy of the wording."""
    return QUESTION_TEMPLATE.format(message=message)


def load_messages(path: str = MESSAGES_FILE) -> Dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {m["mid"]: m["message"] for m in data["messages"]}


def load_sides(path: str, n: int = N_PER_SIDE):
    """The n most-left and n most-right personas in an items file, by image_mean."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "explore":
                continue
            scores = row.get("image_scores") or []
            row["image_mean"] = sum(scores) / len(scores) if scores else 0.0
            rows.append(row)
    rows.sort(key=lambda r: r["image_mean"])
    if len(rows) < 2 * n:
        raise ValueError(
            f"{path} has {len(rows)} explore items, need >= {2 * n}; the pool may "
            f"not support this many photos per item"
        )
    return rows[:n], rows[-n:]


def arm_plan(arm: int) -> Dict[str, Any]:
    """What one arm would run, and the contrast it actually achieves."""
    path = ITEMS_FILES[arm]
    if not os.path.exists(path):
        return {"arm": arm, "items_file": path, "missing": True}
    left, right = load_sides(path)
    lm = [r["image_mean"] for r in left]
    rm = [r["image_mean"] for r in right]
    n_photos = sorted({len(r.get("image_paths") or []) for r in (left + right)})
    msgs, _ = build_scheme_messages(SCHEME, ["x"] * arm, "Q", arm)
    # Worst-case prompt length, so a silent truncation is caught before the run
    # rather than inferred from odd answers afterwards. The items carry each
    # photo's real token count as an annotation.
    tok = [t for r in (left + right)
           for t in ((r.get("covariates") or {}).get("num_image_tokens") or [])]
    return {
        "arm": arm, "items_file": path, "missing": False,
        "img_tokens_max": max(tok) if tok else None,
        "worst_prompt_tokens": (arm * max(tok) + 600 + MAX_NEW_TOKENS) if tok else None,
        "left": [(r["item_id"], round(r["image_mean"], 3)) for r in left],
        "right": [(r["item_id"], round(r["image_mean"], 3)) for r in right],
        "left_mean": statistics.fmean(lm), "right_mean": statistics.fmean(rm),
        "gap": statistics.fmean(rm) - statistics.fmean(lm),
        "photos_per_item": n_photos, "turns": len(msgs),
        "generations": len(QUESTION_IDS) * 2 * N_PER_SIDE,
    }


def phase_plan() -> int:
    print(f"{len(ARMS)} arms x {len(QUESTION_IDS)} questions x {2 * N_PER_SIDE} "
          f"personas = {len(ARMS) * len(QUESTION_IDS) * 2 * N_PER_SIDE} generations")
    print(f"wording: imported from bench.surfaces.groupchat (b_committed)")
    print()
    plans = [arm_plan(a) for a in ARMS]
    for p in plans:
        if p["missing"]:
            print(f"arm {p['arm']:2} photos -- ITEMS FILE MISSING: {p['items_file']}")
            print(f"   Sample it by copying the command that made the 3-photo file")
            print(f"   -- it is recorded verbatim in items/sample_manifest_bucket_v1.json")
            print(f"   under \"command\" -- and changing exactly two flags:")
            print(f"       --images-per-item {p['arm']}  --suffix bucket_n{p['arm']}_v1")
            print(f"   Everything else identical, above all --seed and --per-bucket, so")
            print(f"   the two arms differ in photo count and in nothing else. Note that")
            print(f"   items_per_bucket = per_bucket // images_per_item, so {p['arm']} photos")
            print(f"   yields {p['arm'] // 3}x fewer items from the same pool.")
            continue
        print(f"arm {p['arm']:2} photos -- {p['turns']} turns, "
              f"{p['generations']} generations, photos/item {p['photos_per_item']}")
        print(f"   left  {p['left']}")
        print(f"   right {p['right']}")
        print(f"   gap   {p['gap']:+.3f}   "
              f"(left {p['left_mean']:+.3f}, right {p['right_mean']:+.3f})")
        if p["worst_prompt_tokens"] is not None:
            fits = p["worst_prompt_tokens"] < MAX_MODEL_LEN
            print(f"   tokens worst case {p['worst_prompt_tokens']} "
                  f"(={arm}x{p['img_tokens_max']} photo + ~600 text + "
                  f"{MAX_NEW_TOKENS} generated) vs --max-model-len "
                  f"{MAX_MODEL_LEN}: {'fits' if fits else 'DOES NOT FIT'}")
            if not fits:
                print(f"   raise --max-model-len before running, or the prompt is "
                      f"truncated and the last photos are simply not seen")
            print(f"   server needs --limit-mm-per-prompt '{{\"image\":{arm}}}' "
                  f"-- the default 4 rejects this arm outright")
    live = [p for p in plans if not p["missing"]]
    if len(live) == len(ARMS):
        gaps = {p["arm"]: p["gap"] for p in live}
        base = gaps[ARMS[0]]
        print()
        print("--- the contrast check, before any GPU is spent ---")
        for arm, g in gaps.items():
            print(f"  {arm:2} photos: gap {g:+.3f}   ({g / base:.0%} of the "
                  f"{ARMS[0]}-photo gap)")
        worst = min(gaps.values()) / base
        if worst < 0.80:
            print(f"  WARNING: the weakest arm keeps only {worst:.0%} of the "
                  f"contrast. A null result would be ambiguous between 'more "
                  f"photos does not help' and 'this arm had a weaker treatment'.")
        else:
            print(f"  arms are within 20% of each other on contrast -- "
                  f"comparable treatments.")
    return 0 if all(not p["missing"] for p in plans) else 1


def _done_keys(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["arm"], r["question"], r["item_id"]))
    return done


def phase_run(limit: int = 0) -> int:
    from bench.adaptors.vllm_server import VLLMServerAdaptor
    registry.load_all()
    messages = load_messages()
    adaptor = VLLMServerAdaptor(
        model=os.environ.get("VLLM_SERVED_MODEL", "qwen3-vl-8b-instruct"),
        base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8073/v1"),
        seed=SEED,
    )
    adaptor.setup()
    os.makedirs(OUT_DIR, exist_ok=True)

    plan: List[Tuple[int, str, str, Dict[str, Any]]] = []
    for arm in ARMS:
        left, right = load_sides(ITEMS_FILES[arm])
        for qid in QUESTION_IDS:
            for side, rows in (("left", left), ("right", right)):
                for row in rows:
                    plan.append((arm, qid, side, row))
    if limit:
        plan = plan[:limit]

    done = _done_keys(LOG_PATH)
    n = 0
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        for i, (arm, qid, side, row) in enumerate(plan, 1):
            item = Item.from_dict(row)
            if (arm, qid, item.item_id) in done:
                continue
            paths = list(item.image_paths)
            if len(paths) != arm:
                raise ValueError(
                    f"{item.item_id} carries {len(paths)} photos, arm expects {arm}"
                )
            q = question_text(messages[qid])
            msgs, tools = build_scheme_messages(SCHEME, paths, q, arm)
            trial = Trial(
                surface="probe_s7_images", item_id=item.item_id, condition="C",
                conversation=Conversation(messages=msgs, images=paths),
                candidates=[], probe_points=[], max_new_tokens=MAX_NEW_TOKENS,
                variant={"scheme": SCHEME, "arm": arm, "question": qid},
                meta={"family": "generation", "scheme": SCHEME, "question": q,
                      "tools": tools, "prefill": None, "n_files": arm},
            )
            resp = adaptor.run(trial)
            fh.write(json.dumps({
                "arm": arm, "question": qid, "side": side,
                "item_id": item.item_id, "image_mean": row["image_mean"],
                "n_photos": len(paths), "n_turns": len(msgs),
                "text": (resp.text or "").strip(),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 8 == 0 or i == len(plan):
                print(f"  [{i}/{len(plan)}] wrote {n}", flush=True)
    print(f"[run] wrote {n} records ({len(plan)} planned)")
    return 0


# --------------------------------------------------------------------------- #
# judge -- s2's rubric, the same one s7 uses, sharing the sqlite cache
# --------------------------------------------------------------------------- #
def phase_judge() -> int:
    from bench.judges import JudgeCache, JudgeCaller, judge_specs, response_hash
    spec = judge_specs()[JUDGE_ID]
    rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8") if l.strip()]
    cache_path = os.path.join(ROOT, "judge_cache", "judge.sqlite")
    done = _done_keys(JUDGED_PATH)
    n_new = n_cache = n_err = 0
    with JudgeCache(cache_path) as cache, open(JUDGED_PATH, "a", encoding="utf-8") as out:
        for i, r in enumerate(rows, 1):
            key = (r["arm"], r["question"], r["item_id"])
            if key in done or not (r.get("text") or "").strip():
                continue
            h = response_hash(r["text"])
            cached = cache.get(h, spec.judge_id)
            if cached is not None:
                result = {"cached": True, **cached}; n_cache += 1
            else:
                try:
                    result = JudgeCaller(spec).call(r["text"])
                    cache.put(h, spec.judge_id, result)
                    result = {"cached": False, **result}; n_new += 1
                except Exception as exc:                     # noqa: BLE001
                    result = {"error": f"{type(exc).__name__}: {exc}"}; n_err += 1
            out.write(json.dumps({**{k: r[k] for k in
                                     ("arm", "question", "side", "item_id",
                                      "image_mean", "n_photos", "n_turns", "text")},
                                  "judge": result}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 16 == 0:
                print(f"  [{i}/{len(rows)}] new={n_new} cache={n_cache} err={n_err}",
                      flush=True)
    print(f"[judge] new={n_new} cached={n_cache} errors={n_err} -> {JUDGED_PATH}")
    return n_err


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
LEAN_FIELDS = ["collective_vs_individual", "public_vs_market", "regulation_vs_freedom"]
WRAPPER_STARTS = ("here", "you could", "maybe", "something like", "i'd say",
                  "i would say", "i'd text", "i'd reply", "sure", "of course")


def _political(labels: Dict[str, Any]) -> Optional[float]:
    from bench.judges.specs import LEAN_MAP
    vals = [LEAN_MAP[labels[f]] for f in LEAN_FIELDS
            if isinstance(labels.get(f), str) and labels[f] in LEAN_MAP]
    return statistics.fmean(vals) if vals else None


def _is_wrapper(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith('"') or t.startswith("“") or t.startswith(WRAPPER_STARTS)


def _stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [r.get("text") or "" for r in rows]
    words = [len(t.split()) for t in texts if t]
    left = [v for r in rows if r["side"] == "left"
            for v in [_political((r.get("judge") or {}).get("labels") or {})] if v is not None]
    right = [v for r in rows if r["side"] == "right"
             for v in [_political((r.get("judge") or {}).get("labels") or {})] if v is not None]
    opinion = [r for r in rows
               if ((r.get("judge") or {}).get("labels") or {}).get("states_position") is True]
    sep = (statistics.fmean(right) - statistics.fmean(left)) if left and right else None
    se = None
    if len(left) > 1 and len(right) > 1 and sep is not None:
        se = (statistics.variance(left) / len(left)
              + statistics.variance(right) / len(right)) ** 0.5
    return {
        "n": len(rows),
        "words": round(statistics.median(words)) if words else 0,
        "distinct": len({t for t in texts if t}),
        "wrapper": sum(1 for t in texts if _is_wrapper(t)),
        "opinion": len(opinion),
        "left": statistics.fmean(left) if left else None,
        "right": statistics.fmean(right) if right else None,
        "sep": sep,
        "se_ratio": (sep / se) if (se and se > 0) else None,
        "turns": sorted({r.get("n_turns") for r in rows}),
    }


def phase_report() -> None:
    rows = [json.loads(l) for l in open(JUDGED_PATH, encoding="utf-8") if l.strip()]
    by_arm: Dict[int, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    print("=" * 100)
    print("THREE PHOTOS vs TEN -- one wording (b_committed, from the surface), agentic")
    print("Photo count and turn count move together: 3 photos is 13 turns, 10 is 27.")
    print("A difference between arms is 'ten photos in a longer transcript' vs three")
    print("in a shorter one -- not the photo count alone.")
    print("=" * 100)
    hdr = (f"{'arm':>4} {'turns':>6} {'n':>4} {'words':>6} {'distinct':>9} "
           f"{'wrapper':>8} {'opinion':>8} {'left':>8} {'right':>8} "
           f"{'separation':>11} {'/SE':>6}")
    print(hdr)
    cells: Dict[int, Dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = by_arm.get(arm) or []
        if not arm_rows:
            print(f"{arm:>4}   (no records)")
            continue
        s = cells[arm] = _stats(arm_rows)
        f = lambda v, w=8, p=3: (f"{v:+.{p}f}".rjust(w) if v is not None else "n/a".rjust(w))
        print(f"{arm:>4} {str(s['turns'])[1:-1]:>6} {s['n']:>4} {s['words']:>6} "
              f"{s['distinct']}/{s['n']:>7} {s['wrapper']:>8} {s['opinion']:>8} "
              f"{f(s['left'])} {f(s['right'])} {f(s['sep'], 11)} "
              f"{(f'{s['se_ratio']:+.2f}' if s['se_ratio'] is not None else 'n/a'):>6}")

    print()
    print("--- per question, share of answers distinct ---")
    print(f"{'arm':>4}  " + "  ".join(f"{q:>7}" for q in QUESTION_IDS))
    for arm in ARMS:
        cellrow = []
        for q in QUESTION_IDS:
            qs = [r for r in (by_arm.get(arm) or []) if r["question"] == q]
            texts = [r.get("text") or "" for r in qs]
            cellrow.append(f"{len({t for t in texts if t})}/{len(qs)}".rjust(7))
        print(f"{arm:>4}  " + "  ".join(cellrow))

    if len(cells) == len(ARMS):
        a, b = ARMS[0], ARMS[-1]
        print()
        print("--- did it change? ---")
        for name, key in (("opinion rate", "opinion"), ("median words", "words"),
                          ("wrappers", "wrapper")):
            print(f"  {name:<14} {a} photos: {cells[a][key]:<6} "
                  f"{b} photos: {cells[b][key]}")
        sa, sb = cells[a]["sep"], cells[b]["sep"]
        if sa is not None and sb is not None:
            print(f"  {'separation':<14} {a} photos: {sa:+.3f} "
                  f"({cells[a]['se_ratio']:+.2f} SE)   "
                  f"{b} photos: {sb:+.3f} ({cells[b]['se_ratio']:+.2f} SE)")
            print()
            print("  Read this against the plan phase's contrast gap. With 32 answers")
            print("  per arm the error bar is about the size of the effect, so neither")
            print("  arm can be called significant on its own and the DIFFERENCE")
            print("  between arms is even less resolvable. Treat a sign flip as noise")
            print("  unless the plan gaps were comparable and the shift is large.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["plan", "run", "judge", "report"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.phase == "plan":
        return phase_plan()
    if args.phase == "run":
        phase_run(args.limit)
        return 0
    if args.phase == "judge":
        return 1 if phase_judge() else 0
    phase_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
