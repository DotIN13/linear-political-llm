"""Does showing ten photos instead of three change the s7 result?

One wording -- `b_committed`, which round 16 chose -- and one thing varied: how
many photos the persona is built from. **The wording is imported from
`bench.surfaces.groupchat`, never retyped here**, so this probe cannot drift away
from the surface it is testing.

    3 photos  x  4 questions  x  8 personas  =  32
    5 photos  x  4 questions  x  8 personas  =  32
    10 photos x  4 questions  x  8 personas  =  32
                                       total    96 generations

**Three points do not resolve a curve here.** Each group's separation carries an
error bar about the size of the effect, so the honest expectation is that all
three land within noise of each other -- which is itself the finding, and is
worth one cheap group to establish rather than assuming. What three points can
do that two cannot is show whether the middle sits between the ends or outside
them: outside is evidence that noise dominates the axis, and that is a cleaner
statement than "10 was a bit lower than 3".

**Both arms run here.** The 3-photo arm is not taken from round 16, for three
reasons that each on their own would break the comparison: round 16 ran on an
H100 whose numbers are not on the same footing as a fresh server, `generation.py`
has changed since so the measurement fingerprint differs, and 32 answers is too
few for the difference between arms to survive a change of hardware as well.

**What this cannot separate, and it is not a small thing.** In the agentic
scheme every photo is one `view_image` turn, so the transcript grows with the
photo count: **13 turns at 3 photos, 17 at 5, 27 at 10**. Photo count and
transcript length move together and there is no version of this design in which
they do not. So a difference between the groups is "more photos in a longer
transcript" versus fewer in a shorter one -- not the photo count alone. Adding a
third point does not fix this; it adds a third point on a line where both things
vary at once.

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
import collections
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
    5: os.path.join(ROOT, "items", "explore_bucket_n5_v1.jsonl"),
    10: os.path.join(ROOT, "items", "explore_bucket_n10_v1.jsonl"),
}
# Ascending, because ARMS[0] is the reference the contrast ratio is taken against
# and the report prints in this order.
ARMS: Tuple[int, ...] = (3, 5, 10)

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
    msgs, _ = build_scheme_messages(SCHEME, ["x"] * arm, "no_photos", arm)
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
    print(f"{len(ARMS)} photo-count groups ({', '.join(f'{a} photos' for a in ARMS)}) "
          f"x {len(QUESTION_IDS)} questions x {2 * N_PER_SIDE} personas "
          f"(={N_PER_SIDE} most left-looking + {N_PER_SIDE} most right-looking) "
          f"= {len(ARMS) * len(QUESTION_IDS) * 2 * N_PER_SIDE} generations")
    print("wording: the family-group-chat surface's own question, imported from "
          "bench/surfaces/groupchat.py -- never a copy")
    print()
    plans = [arm_plan(a) for a in ARMS]
    for p in plans:
        if p["missing"]:
            print(f"{p['arm']:2} photos per persona -- ITEMS FILE MISSING: "
                  f"{p['items_file']}")
            print(f"   Sample it by copying the command that made the 3-photo file")
            print(f"   -- it is recorded verbatim in items/sample_manifest_bucket_v1.json")
            print(f"   under \"command\" -- and changing exactly two flags:")
            print(f"       --images-per-item {p['arm']}  --suffix bucket_n{p['arm']}_v1")
            print(f"   Everything else identical, above all --seed and --per-bucket, so")
            print(f"   the two arms differ in photo count and in nothing else. Note that")
            print(f"   items_per_bucket = per_bucket // images_per_item, so {p['arm']} photos")
            print(f"   yields {p['arm'] // 3}x fewer items from the same pool.")
            continue
        print(f"{p['arm']:2} photos per persona -- {p['turns']} conversation turns, "
              f"{p['generations']} generations, photos/item {p['photos_per_item']}")
        print(f"   most left-looking  (item id, photo score): {p['left']}")
        print(f"   most right-looking (item id, photo score): {p['right']}")
        print(f"   photo-score gap {p['gap']:+.3f}   (left-looking "
              f"{p['left_mean']:+.3f}, right-looking {p['right_mean']:+.3f}) -- this is"
              f" the treatment, not a judge score")
        if p["worst_prompt_tokens"] is not None:
            fits = p["worst_prompt_tokens"] < MAX_MODEL_LEN
            print(f"   tokens worst case {p['worst_prompt_tokens']} "
                  f"(={p['arm']}x{p['img_tokens_max']} photo + ~600 text + "
                  f"{MAX_NEW_TOKENS} generated) vs --max-model-len "
                  f"{MAX_MODEL_LEN}: {'fits' if fits else 'DOES NOT FIT'}")
            if not fits:
                print(f"   raise --max-model-len before running, or the prompt is "
                      f"truncated and the last photos are simply not seen")
            print(f"   server needs --limit-mm-per-prompt "
                  f"'{{\"image\":{p['arm']}}}' -- the default of 4 rejects a "
                  f"{p['arm']}-photo prompt outright")
    live = [p for p in plans if not p["missing"]]
    if len(live) == len(ARMS):
        gaps = {p["arm"]: p["gap"] for p in live}
        base = gaps[ARMS[0]]
        print()
        print("--- photo-score contrast check (the TREATMENT), before any GPU is spent ---")
        for arm_n, g in gaps.items():
            print(f"  {arm_n:2} photos: photo-score gap {g:+.3f}   "
                  f"({g / base:.0%} of the {ARMS[0]}-photo gap)")
        worst = min(gaps.values()) / base
        if worst < 0.80:
            print(f"  WARNING: the weakest arm keeps only {worst:.0%} of the "
                  f"contrast. A null result would be ambiguous between 'more "
                  f"photos does not help' and 'this arm had a weaker treatment'.")
        else:
            print(f"  the groups are within 20% of each other on photo-score "
                  f"contrast -- comparable treatments, so a null result would be "
                  f"about photo count rather than about a weaker manipulation.")
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
                surface="probe_s7_images", item_id=item.item_id, condition="photos",
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
# Asked from the spec rather than typed, so a rubric change cannot leave a stale
# field list here averaging an axis that no longer exists -- which is how the
# `states_position` bug got in.
def lean_fields() -> List[str]:
    from bench.judges.specs import LEAN_MAP, judge_specs
    spec = judge_specs()[JUDGE_ID]
    return [f for f in spec.fields if spec.label_map.get(f) is LEAN_MAP]
WRAPPER_STARTS = ("here", "you could", "maybe", "something like", "i'd say",
                  "i would say", "i'd text", "i'd reply", "sure", "of course")


def _political(labels: Dict[str, Any]) -> Optional[float]:
    """Mean of whichever lean axes the judge actually filled in.

    **null axes are skipped, not read as centre.** An axis the text never touches
    is absent, and averaging it in as 0 pulls a one-sided answer towards
    moderate -- which attenuated every effect this project has measured.
    """
    from bench.judges.specs import LEAN_MAP
    vals = [LEAN_MAP[labels[f]] for f in lean_fields()
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
    # Use the judge's own field name. Reading a field the rubric does not emit
    # yields 0 for every group, which looks exactly like "the model stated no
    # position in any answer" -- a null that is really a typo. A test asserts
    # every judge field touched here is one the spec declares.
    opinion = [r for r in rows
               if ((r.get("judge") or {}).get("labels") or {})
               .get("political_content_present") is True]
    refusals = [r for r in rows
                if ((r.get("judge") or {}).get("labels") or {}).get("refusal") is True]
    sep = (statistics.fmean(right) - statistics.fmean(left)) if left and right else None
    se = None
    if len(left) > 1 and len(right) > 1 and sep is not None:
        # Sample variance (n-1), the unbiased estimator for the SE of a mean.
        # Round 16's bake-off used population variance (n), so its /SE figures
        # are inflated by sqrt(n/(n-1)) -- about 3% at n=16. /SE is the only
        # figure comparable across rounds, but only once the estimator matches,
        # so round 16's +0.85 is about +0.82 on this scale.
        se = (statistics.variance(left) / len(left)
              + statistics.variance(right) / len(right)) ** 0.5
    return {
        "n": len(rows),
        "words": round(statistics.median(words)) if words else 0,
        "distinct": len({t for t in texts if t}),
        "wrapper": sum(1 for t in texts if _is_wrapper(t)),
        "opinion": len(opinion),
        "refusal": len(refusals),
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
    print(f"PHOTOS PER PERSONA: {' vs '.join(str(a) for a in ARMS)} -- one wording "
          f"(the surface's own), tool-using scheme")
    print()
    print("Two different scales below, both signed decimals near zero -- do not compare")
    print("their magnitudes:")
    print("  photo-score gap  = the TREATMENT. Mean photo slant of the right-looking")
    print("                     personas minus the left-looking ones. Printed by `plan`.")
    print("  judge separation = the RESPONSE. Mean judge lean of answers to right-looking")
    print("                     personas minus left-looking ones. Printed below.")
    print("  /SE              = judge separation over its own error bar. Dimensionless,")
    print("                     so this is the only column comparable across rounds.")
    print()
    turns_by_arm = {a: len(build_scheme_messages(SCHEME, ["x"] * a, "no_photos", a)[0])
                    for a in ARMS}
    print("Photo count and turn count move together -- every photo is one more")
    print("image-opening turn: " + ", ".join(f"{a} photos = {t} turns"
                                            for a, t in turns_by_arm.items()) + ".")
    print("So a difference between the groups is 'more photos in a longer transcript'")
    print("versus fewer in a shorter one -- not the photo count alone.")
    print("=" * 100)
    hdr = (f"{'photos':>7} {'turns':>6} {'n':>4} {'words':>6} {'distinct':>9} "
           f"{'wrapper':>8} {'opinion':>8} {'refusal':>8} {'judge L':>8} {'judge R':>8} "
           f"{'judge sep':>11} {'/SE':>6}")
    print(hdr)
    cells: Dict[int, Dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = by_arm.get(arm) or []
        if not arm_rows:
            print(f"{arm:>7}   (no photo-count group on record)")
            continue
        s = cells[arm] = _stats(arm_rows)
        f = lambda v, w=8, p=3: (f"{v:+.{p}f}".rjust(w) if v is not None else "n/a".rjust(w))
        print(f"{arm:>7} {str(s['turns'])[1:-1]:>6} {s['n']:>4} {s['words']:>6} "
              f"{s['distinct']}/{s['n']:>7} {s['wrapper']:>8} {s['opinion']:>8} "
              f"{s['refusal']:>8} "
              f"{f(s['left'])} {f(s['right'])} {f(s['sep'], 11)} "
              f"{(f'{s['se_ratio']:+.2f}' if s['se_ratio'] is not None else 'n/a'):>6}")

    print()
    print("--- per question, share of answers distinct ---")
    print(f"{'photos':>7}  " + "  ".join(f"{q:>7}" for q in QUESTION_IDS))
    for arm in ARMS:
        cellrow = []
        for q in QUESTION_IDS:
            qs = [r for r in (by_arm.get(arm) or []) if r["question"] == q]
            texts = [r.get("text") or "" for r in qs]
            cellrow.append(f"{len({t for t in texts if t})}/{len(qs)}".rjust(7))
        print(f"{arm:>7}  " + "  ".join(cellrow))

    # --- pooled across groups -------------------------------------------------
    # The `distinct` column above is computed inside one photo-count group, so a
    # near-duplicate in another group is invisible to it. Round 17 showed that
    # mattering: the 3-photo and 10-photo left-looking answers to m01 differed by
    # two words, from different personas with seven more photos, while both groups
    # scored 31/32 distinct. A dose curve over photo count is not worth reading if
    # the answers barely respond to the persona at all, so this block is a
    # correctness guard on the numbers above rather than a separate measure.
    print()
    print("--- distinctness POOLED across photo-count groups (the guard) ---")
    print("    the `distinct` column above only looks inside one group; this looks")
    print("    across all of them, per question, which is where a shared opener hides")
    for q in QUESTION_IDS:
        qs = [r for r in rows if r["question"] == q]
        texts = [r.get("text") or "" for r in qs]
        pooled = len({t for t in texts if t})
        within = sum(len({(r.get("text") or "") for r in qs if r["arm"] == a and r.get("text")})
                     for a in ARMS)
        print(f"  {q}: pooled {pooled}/{len(qs)} distinct"
              f"   vs {within}/{len(qs)} summed within groups"
              f"{'   <-- duplicates ACROSS groups' if pooled < within else ''}")
    all_texts = [r.get("text") or "" for r in rows if r.get("text")]
    print(f"  all questions: pooled {len(set(all_texts))}/{len(all_texts)} distinct")
    # The shared-opener count is the round-15 diagnostic: at short lengths the
    # opener is most of the answer, so few openers means little room for a persona.
    openers = collections.Counter(" ".join(t.split()[:8]) for t in all_texts)
    print(f"  distinct 8-word openers: {len(openers)} across {len(all_texts)} answers")
    for opener, count in openers.most_common(3):
        if count > 1:
            print(f"    {count:>3}x  \"{opener}...\"")

    if len(cells) == len(ARMS):
        a, b = ARMS[0], ARMS[-1]
        print()
        print("--- did it change? ---")
        for name, key in (("opinion rate (n)", "opinion"),
                          ("refusals (n)", "refusal"),
                          ("median words", "words"),
                          ("wrappers (n)", "wrapper")):
            row = "   ".join(f"{arm_n} photos: {cells[arm_n][key]}" for arm_n in ARMS)
            print(f"  {name:<18} {row}")
        sa, sb = cells[a]["sep"], cells[b]["sep"]
        if sa is not None and sb is not None:
            def _se(arm_key: int) -> str:
                # None when a group has no spread at all -- which happens on
                # synthetic data and would otherwise crash the whole report.
                r = cells[arm_key]["se_ratio"]
                return f"{r:+.2f} SE" if r is not None else "SE undefined (no spread)"
            row = "   ".join(
                f"{arm_n} photos: {cells[arm_n]['sep']:+.3f} ({_se(arm_n)})"
                for arm_n in ARMS if cells[arm_n]["sep"] is not None)
            print(f"  {'judge separation':<18} {row}")
            # Is the middle group between the ends, or outside them? Outside is
            # evidence that noise dominates this axis rather than a dose curve.
            seps = [cells[arm_n]["sep"] for arm_n in ARMS
                    if cells[arm_n]["sep"] is not None]
            if len(seps) >= 3:
                mid = seps[1:-1]
                lo, hi = min(seps[0], seps[-1]), max(seps[0], seps[-1])
                outside = [v for v in mid if not lo <= v <= hi]
                if outside:
                    print(f"  -> the middle group falls OUTSIDE the two ends "
                          f"({outside}): not a dose curve, noise dominating")
                else:
                    print(f"  -> the middle group falls between the ends: consistent "
                          f"with a weak monotone trend, but see the error bars")
            print()
            n_each = cells[ARMS[0]]["n"]
            print(f"  Read this against the plan phase's photo-score contrast gaps.")
            print(f"  With {n_each} answers per photo-count group the error bar is about")
            print(f"  the size of the effect, so no group is significant on its own and")
            print(f"  the DIFFERENCES between groups are less resolvable still. Treat a")
            print(f"  sign flip or a non-monotone middle as noise unless the contrast")
            print(f"  gaps were comparable and the shift is large.")


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
