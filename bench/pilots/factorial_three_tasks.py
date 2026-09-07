"""The three tasks under one instrument: news ranking, family chat, letter.

    3 tasks x 2 conversations x 12 items x 18 personas  = 1296   photos
    3 tasks x 12 items                                  =   36   no_photos
    the letter under agentic, one band, read twice      =   72
                                                          -----
                                                          1404

**The twelve items line up across all three tasks**, which is what this design
exists to exploit. The family chat and the letter carry the same twelve topics in
the same order; the news pool was rebuilt to those same twelve. So topic is held
fixed while task varies, and "does the task shape matter" can be asked without
confounding it with subject matter.

**The news ranking has one question, not twelve -- so its twelve are deals.** Its
pool holds 24 candidates, two sides per topic, and a trial shows 12: one side of
each, balanced six left and six right. Twelve seeds give twelve independent deals,
so every task contributes twelve measurements per persona and the three are the
same size rather than one being a rounding error beside the others.

**The baseline runs once per question.** `no_photos` is the task on its own -- no
share line, no memory directories, no filenames, no tool calls. It has no persona
in it, so it is identical across all eighteen; and no scaffolding, so it is
identical across both conversation schemes. The surfaces declare both invariances
and this pilot asks them rather than assuming.

**What this design cannot separate**, written here rather than discovered later:

* Conversation is tangled with turn count -- 5 turns against 13 (7 against 15 for
  the letter). "It came to know you by opening your files" and "there was more
  conversation first" are one thing here.
* Photo slant is tangled with scene content: right-looking photos skew rural and
  open, left-looking urban and walkable.
* The agentic transcript is a fiction we wrote. The model never chose to look.

    python -m bench.pilots.factorial_three_tasks plan
    python -m bench.pilots.factorial_three_tasks run
    python -m bench.pilots.factorial_three_tasks judge
    python -m bench.pilots.factorial_three_tasks report
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
from bench.types import Conversation, Item, Trial, baseline_item

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "runs", "factorial_three_tasks")
LOG_PATH = os.path.join(OUT_DIR, "log.jsonl")
JUDGED_PATH = os.path.join(OUT_DIR, "judged.jsonl")
ITEMS_FILE = os.path.join(ROOT, "items", "explore_bucket_v1.jsonl")

TASKS = ("s3_digest", "s7_family_chat", "s8_letter_answered")
SCHEMES = ("chat", "agentic")
BANDS = ("low", "mid", "high")
PER_BAND = 6
N_ITEMS = 12                    # questions, or deals for the news ranking
SEED = 20260907
# One cell read twice, for the repeatability diagnostic: the letter under
# agentic, **one band only**. Repeating all three bands would cost 216
# generations for a number that 72 measures just as well -- repeat readings
# contribute about 2.5% of the error, so this is a diagnostic, not precision.
REPEAT_CELL = ("s8_letter_answered", "agentic")
REPEAT_BAND = "low"

# The news ranking is scored without a judge; the other two share s2's rubric.
JUDGE_BY_TASK = {"s7_family_chat": "s2_proposal", "s8_letter_answered": "s2_proposal"}


def load_personas(path: Optional[str] = None, per_band: int = PER_BAND):
    """The `per_band` most extreme explore items in each bucket, by photo score.

    `path` defaults to the module global **read at call time**, not bound as a
    default argument -- a default is captured at import, so pointing ITEMS_FILE
    somewhere else afterwards would have no effect and the function would keep
    reading the original file while appearing to have been redirected.
    """
    path = path or ITEMS_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The stimulus items live on the cluster, not in the "
            f"repo. Run this where they are, or point ITEMS_FILE at a copy.")
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
    by_band: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_band[r.get("bucket", "mid")].append(r)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for band in BANDS:
        pool = sorted(by_band.get(band, []), key=lambda r: r["image_mean"])
        if len(pool) < per_band:
            raise ValueError(f"bucket {band!r} has {len(pool)} explore items, "
                             f"need >= {per_band}")
        # most extreme *away from centre*: the lowest of low, the highest of high,
        # and the closest to zero for the middle band.
        if band == "low":
            out[band] = pool[:per_band]
        elif band == "high":
            out[band] = pool[-per_band:]
        else:
            out[band] = sorted(pool, key=lambda r: abs(r["image_mean"]))[:per_band]
    return out


def task_items(task: str) -> List[str]:
    """The twelve. Question ids for most tasks; deal seeds for the news ranking."""
    surface = registry.get_surface(task)()
    qids = surface.question_ids()
    if len(qids) >= N_ITEMS:
        return qids[:N_ITEMS]
    # one question, so the twelve are deals: one per seed
    return [f"deal{i:02}" for i in range(1, N_ITEMS + 1)]


def item_seed(task: str, item: str) -> int:
    """A deal id carries its own seed; a question id does not vary the deal."""
    if item.startswith("deal"):
        return SEED + int(item[4:])
    return SEED


def variant_for(task: str, scheme: str, item: str) -> Dict[str, Any]:
    surface = registry.get_surface(task)()
    qids = surface.question_ids()
    qid = item if item in qids else qids[0]
    return {"scheme": scheme, "question": qid}


def plan_rows() -> List[Dict[str, Any]]:
    """Every trial this pilot will run, as plain dicts."""
    personas = load_personas()
    rows: List[Dict[str, Any]] = []
    for task in TASKS:
        surface = registry.get_surface(task)()
        items = task_items(task)
        for scheme in SCHEMES:
            for item in items:
                for band in BANDS:
                    for p in personas[band]:
                        reps = 2 if ((task, scheme) == REPEAT_CELL
                                     and band == REPEAT_BAND) else 1
                        for rep in range(1, reps + 1):
                            rows.append({"task": task, "scheme": scheme, "item": item,
                                         "band": band, "item_id": p["item_id"],
                                         "image_mean": p["image_mean"],
                                         "condition": "photos", "rep": rep})
        # the baseline: once per item. Ask the surface whether the scheme matters.
        scheme_free = surface.is_scheme_invariant("no_photos")
        base_schemes = (SCHEMES[0],) if scheme_free else SCHEMES
        for item in items:
            for scheme in base_schemes:
                rows.append({"task": task, "scheme": scheme, "item": item,
                             "band": None, "item_id": "__baseline__",
                             "image_mean": None, "condition": "no_photos", "rep": 1})
    return rows


def phase_plan() -> int:
    registry.load_all()
    try:
        personas = load_personas()
    except FileNotFoundError as exc:
        print(f"ITEMS FILE MISSING: {exc}")
        return 1
    rows = plan_rows()
    print(f"{len(TASKS)} tasks ({', '.join(TASKS)})")
    print(f"x {len(SCHEMES)} conversations ({', '.join(SCHEMES)})")
    print(f"x {N_ITEMS} items      questions, or deals for the news ranking")
    print(f"x {len(BANDS) * PER_BAND} personas  ({PER_BAND} per band x {len(BANDS)} bands)")
    print()
    by = collections.Counter((r["task"], r["condition"]) for r in rows)
    for task in TASKS:
        surface = registry.get_surface(task)()
        print(f"  {task:20} photos {by[(task,'photos')]:>5}   "
              f"no_photos {by[(task,'no_photos')]:>3}   "
              f"judge {JUDGE_BY_TASK.get(task) or 'none -- deterministic'}")
    reps = sum(1 for r in rows if r["rep"] == 2)
    print()
    print(f"  repeat readings ({REPEAT_CELL[0]} / {REPEAT_CELL[1]}, "
          f"{REPEAT_BAND} band only): {reps}")
    print(f"  TOTAL GENERATIONS: {len(rows)}")
    print(f"  judge calls: {sum(1 for r in rows if r['task'] in JUDGE_BY_TASK)}")
    print()
    print("--- photo-score contrast (the TREATMENT, not a judge score) ---")
    means = {}
    for band in BANDS:
        m = [p["image_mean"] for p in personas[band]]
        means[band] = statistics.fmean(m)
        print(f"  {band:5} n={len(m)}  mean {means[band]:+.3f}   "
              f"range {min(m):+.3f} .. {max(m):+.3f}")
    gap = means["high"] - means["low"]
    print(f"  low -> high gap {gap:+.3f}")
    ordered = means["low"] < means["mid"] < means["high"]
    print(f"  bands ordered low < mid < high: {ordered}")
    if not ordered:
        print("  WARNING: the bands are not ordered on the treatment; stop and report")
        return 1
    print()
    print("--- turn counts, so the conversation confound is on the record ---")
    it = baseline_item()
    for task in TASKS:
        surface = registry.get_surface(task)()
        row = {}
        for scheme in SCHEMES:
            v = variant_for(task, scheme, task_items(task)[0])
            n = len(surface.build(it, "no_photos", dict(v), seed=SEED).conversation.messages)
            p = load_personas()[BANDS[0]][0]
            nc = len(surface.build(Item.from_dict(p), "photos", dict(v),
                                   seed=SEED).conversation.messages)
            row[scheme] = (nc, n)
        print(f"  {task:20} " + "   ".join(
            f"{s}: photos {row[s][0]:>2} / no_photos {row[s][1]:>2}" for s in SCHEMES))
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _key(r: Dict[str, Any]) -> Tuple:
    return (r["task"], r["scheme"], r["item"], r["item_id"], r["condition"], r["rep"])


def _done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(_key(json.loads(line)))
    return done


def phase_run(limit: int = 0) -> int:
    from bench.adaptors.vllm_server import VLLMServerAdaptor
    registry.load_all()
    personas = load_personas()
    by_id = {p["item_id"]: p for band in BANDS for p in personas[band]}
    adaptor = VLLMServerAdaptor(
        model=os.environ.get("VLLM_SERVED_MODEL", "qwen3-vl-8b-instruct"),
        base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8073/v1"),
        seed=SEED,
    )
    adaptor.setup()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = plan_rows()
    if limit:
        rows = rows[:limit]
    done = _done(LOG_PATH)
    n = 0
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            if _key(r) in done:
                continue
            surface = registry.get_surface(r["task"])()
            item = baseline_item() if r["item_id"] == "__baseline__" \
                else Item.from_dict(by_id[r["item_id"]])
            variant = variant_for(r["task"], r["scheme"], r["item"])
            # the repeat reading must not dedup against the first one
            if r["rep"] > 1:
                variant["rep"] = r["rep"]
            trial = surface.build(item, r["condition"], variant,
                                  seed=item_seed(r["task"], r["item"]))
            resp = adaptor.run(trial)
            out = {**r, "text": (resp.text or "").strip(),
                   "n_turns": len(trial.conversation.messages),
                   "n_images": trial.meta.get("n_images"),
                   "error": resp.error}
            # the news ranking is scored here, with no judge: which five it picked
            if r["task"] == "s3_digest":
                extracted = surface.extract(resp, trial)
                picks = getattr(extracted, "extra", None) or {}
                for f in ("picked_hids", "n_picked", "parse_ok", "slant_rel_mean",
                          "slant_shown_mean", "picked_slant_mean", "n_right",
                          "n_right_shown", "n_shown", "topics_covered"):
                    if f in picks:
                        out[f] = picks[f]
                out["order"] = trial.variant.get("order")
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0 or resp.error:
                print(f"  [{i}/{len(rows)}] wrote {n}  err={resp.error}", flush=True)
    print(f"[run] wrote {n} records ({len(rows)} planned)")
    return 0


# --------------------------------------------------------------------------- #
# judge -- only the two tasks that need one
# --------------------------------------------------------------------------- #
def phase_judge() -> int:
    from bench.judges import JudgeCache, JudgeCaller, judge_specs, response_hash
    specs = {t: judge_specs()[j] for t, j in JUDGE_BY_TASK.items()}
    rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["task"] in specs and (r.get("text") or "").strip()]
    cache_path = os.path.join(ROOT, "judge_cache", "judge.sqlite")
    done = _done(JUDGED_PATH)
    new = cached = errors = 0
    with JudgeCache(cache_path) as cache, open(JUDGED_PATH, "a", encoding="utf-8") as out:
        for i, r in enumerate(rows, 1):
            if _key(r) in done:
                continue
            spec = specs[r["task"]]
            h = response_hash(r["text"])
            hit = cache.get(h, spec.judge_id)
            if hit is not None:
                result = {"cached": True, **hit}; cached += 1
            else:
                try:
                    result = JudgeCaller(spec).call(r["text"])
                    cache.put(h, spec.judge_id, result)
                    result = {"cached": False, **result}; new += 1
                except Exception as exc:                      # noqa: BLE001
                    result = {"error": f"{type(exc).__name__}: {exc}"}; errors += 1
            out.write(json.dumps({**r, "judge": result}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 100 == 0:
                print(f"  [{i}/{len(rows)}] new={new} cached={cached} err={errors}",
                      flush=True)
    print(f"[judge] new={new} cached={cached} errors={errors} -> {JUDGED_PATH}")
    return errors


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _political(labels: Dict[str, Any]) -> Optional[float]:
    """Mean of whichever lean axes the judge filled in. Nulls are skipped, not
    read as centre -- an axis the text never touches is absent, and averaging it
    in as 0 pulls a one-sided answer towards moderate."""
    from bench.judges.specs import LEAN_MAP, judge_specs
    spec = judge_specs()["s2_proposal"]
    fields = [f for f in spec.fields if spec.label_map.get(f) is LEAN_MAP]
    vals = [LEAN_MAP[labels[f]] for f in fields
            if isinstance(labels.get(f), str) and labels[f] in LEAN_MAP]
    return statistics.fmean(vals) if vals else None


def _band_stats(vals_by_band: Dict[str, List[float]]) -> Dict[str, Any]:
    have = {b: v for b, v in vals_by_band.items() if v}
    if "low" not in have or "high" not in have:
        return {}
    lo, hi = have["low"], have["high"]
    sep = statistics.fmean(hi) - statistics.fmean(lo)
    se = None
    if len(lo) > 1 and len(hi) > 1:
        se = (statistics.variance(lo) / len(lo)
              + statistics.variance(hi) / len(hi)) ** 0.5
    mid = statistics.fmean(have["mid"]) if have.get("mid") else None
    monotone = None
    if mid is not None:
        monotone = statistics.fmean(lo) <= mid <= statistics.fmean(hi)
    return {"sep": sep, "se_ratio": (sep / se) if se else None,
            "low": statistics.fmean(lo), "mid": mid, "high": statistics.fmean(hi),
            "n_low": len(lo), "n_high": len(hi), "monotone": monotone}


def phase_report() -> None:
    print("=" * 100)
    print("THREE TASKS, ONE INSTRUMENT")
    print()
    print("Two scales below, both signed decimals near zero. Do not compare their")
    print("magnitudes:")
    print("  photo-score gap  = the TREATMENT -- how far apart the persona bands are")
    print("                     in photo slant. Printed by `plan`.")
    print("  band separation  = the RESPONSE -- how far apart what the model wrote is.")
    print("                     For the news ranking that is the slant of the outlets")
    print("                     it chose, needing no judge; for the other two it is the")
    print("                     judge's lean axes.")
    print("  /SE              = separation over its own error bar. Dimensionless, so")
    print("                     this is the only column comparable across rounds.")
    print()
    print("Conversation and turn count move together (5 turns against 13, 7 against")
    print("15 for the letter), so any chat-versus-agentic difference is 'more")
    print("conversation' and 'opened your files' at once.")
    print("=" * 100)

    logs = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8") if l.strip()]
    judged = {}
    if os.path.exists(JUDGED_PATH):
        for l in open(JUDGED_PATH, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                judged[_key(r)] = r

    hdr = (f"{'task':22}{'conversation':>14}{'n':>6}{'answers':>9}{'refusals':>10}"
           f"{'band low':>10}{'mid':>9}{'high':>9}{'separation':>12}{'/SE':>7}")
    print(hdr)
    for task in TASKS:
        for scheme in SCHEMES:
            cell = [r for r in logs if r["task"] == task and r["scheme"] == scheme
                    and r["condition"] == "photos"]
            if not cell:
                print(f"{task:22}{scheme:>14}   (no records)")
                continue
            vals: Dict[str, List[float]] = {b: [] for b in BANDS}
            refusals = 0
            for r in cell:
                if task == "s3_digest":
                    v = r.get("slant_rel_mean")
                else:
                    j = judged.get(_key(r), {}).get("judge") or {}
                    labels = j.get("labels") or {}
                    if labels.get("refusal") is True:
                        refusals += 1
                    v = _political(labels) if labels else None
                if v is not None and r.get("band"):
                    vals[r["band"]].append(v)
            st = _band_stats(vals)
            f = lambda x, w=9: (f"{x:+.3f}".rjust(w) if x is not None else "n/a".rjust(w))
            answered = sum(1 for r in cell if (r.get("text") or "").strip())
            print(f"{task:22}{scheme:>14}{len(cell):>6}{answered:>9}{refusals:>10}"
                  f"{f(st.get('low'), 10)}{f(st.get('mid'))}{f(st.get('high'))}"
                  f"{f(st.get('sep'), 12)}"
                  f"{(f'{st['se_ratio']:+.2f}' if st.get('se_ratio') else 'n/a'):>7}")
            if st.get("monotone") is False:
                print(f"{'':36}-> the middle band is OUTSIDE the two ends: not a dose,"
                      f" noise dominating")

    # the baseline, and what the persona apparatus is worth against it
    print()
    print("--- the no-photo baseline: what the model says if you simply ask ---")
    for task in TASKS:
        base = [r for r in logs if r["task"] == task and r["condition"] == "no_photos"]
        if not base:
            continue
        vals = []
        for r in base:
            if task == "s3_digest":
                v = r.get("slant_rel_mean")
            else:
                j = judged.get(_key(r), {}).get("judge") or {}
                v = _political(j.get("labels") or {}) if j.get("labels") else None
            if v is not None:
                vals.append(v)
        shown = [r for r in logs if r["task"] == task and r["condition"] == "photos"]
        sv = []
        for r in shown:
            v = r.get("slant_rel_mean") if task == "s3_digest" else (
                _political((judged.get(_key(r), {}).get("judge") or {}).get("labels") or {}))
            if v is not None:
                sv.append(v)
        if vals:
            b = statistics.fmean(vals)
            line = f"  {task:22} baseline {b:+.3f}  (n={len(vals)})"
            if sv:
                line += (f"   with photos {statistics.fmean(sv):+.3f}"
                         f"   difference {statistics.fmean(sv) - b:+.3f}")
            print(line)
    print()
    print("  The difference is what the whole persona apparatus is worth -- photos and")
    print("  framing together. It is not the photo effect on its own: nothing here")
    print("  separates the pixels from the sentence that introduced them.")

    # repeatability
    print()
    print("--- repeatability, from the cell read twice ---")
    pairs = collections.defaultdict(dict)
    for r in logs:
        if (r["condition"] == "photos" and (r["task"], r["scheme"]) == REPEAT_CELL
                and r.get("band") == REPEAT_BAND):
            pairs[(r["item"], r["item_id"])][r["rep"]] = (r.get("text") or "").strip()
    both = [v for v in pairs.values() if 1 in v and 2 in v]
    if both:
        same = sum(1 for v in both if v[1] == v[2])
        print(f"  {REPEAT_CELL[0]} / {REPEAT_CELL[1]} ({REPEAT_BAND}): "
              f"{same}/{len(both)} cells "
              f"reproduced their own answer byte for byte ({same/len(both):.0%})")
    else:
        print("  no repeat pairs on record yet")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["plan", "run", "judge", "report"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.phase == "plan":
        return phase_plan()
    if args.phase == "run":
        return phase_run(args.limit)
    if args.phase == "judge":
        return 1 if phase_judge() else 0
    phase_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
