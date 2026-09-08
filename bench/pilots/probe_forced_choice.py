"""Pilot: do the new forced-choice surfaces move with the persona?

Three tasks from `bench/surfaces/tasks/`, chosen as Wren's top three:

  s9_neighborhood   pick 3 of 10   DV = mean right_c of the picks
  s12_explain       pick 3 of 8    DV = mean right_c, twice: a *live* topic
                                   (inflation) and an *inert* one (sky_blue)
  s13_patch         pick 1 of 4    NOT political -- the negative control. DV is
                                   accuracy, and whether the answer moves at all.

This lives in `bench/pilots/`, which is outside MEASUREMENT_GLOBS, so nothing here
touches the identity of any trial. It builds prompts from the task packages' own
prompt files, so the *material* is the committed material; only the assembly and
the readers are drafts. If the pilot shows something, these become real surfaces.

ORDER CONTROL -- the one thing this does differently from round 9
-----------------------------------------------------------------
Round 9 called `sampled_order`, which seeds on item_id, so **each bucket drew its
own set of presentation orders**. Position bias in the s3 task ran from 0.889 at
position 1 to 0.139 at position 10; letting a seed decide which bucket gets which
order pushes a bias that large straight into the between-bucket contrast as
variance.

Reversal does not fix it either. Pairing an order with its exact reverse cancels
the anti-symmetric part of the position-weight function -- the "earlier is
better" trend -- and leaves the symmetric part untouched, so a U-shaped bias
(primacy *and* recency) survives it completely.

So: do not try to remove position bias, make it **identical across the buckets
being compared**. Every bucket gets the same set of rotations of the same base
order, {0..5} x {fwd, rev}. Position bias then lands as one constant offset on
every bucket mean and cancels exactly in any bucket difference -- no variance
from that source, and no extra trials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench.surfaces.shared.prompts import pool, text as prompt_text
from bench.surfaces.shared.transcript import build_scheme_messages
from bench.types import Conversation, Item, Trial

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = os.path.join(ROOT, "bench", "surfaces", "tasks")
ITEMS_FILE = os.path.join(ROOT, "items", "explore_bucket_v1.jsonl")
OUT_DIR = os.path.join(ROOT, "runs", "probe_forced_choice")
TRIALS_PATH = os.path.join(OUT_DIR, "trials.jsonl")

BUCKETS = ("low", "mid", "high")
ITEMS_PER_BUCKET = 6
SEED = 20260908

# Parsing free prose is how round 9 manufactured a fake result: a keyword matcher
# read the examples inside the model's own clarifying question as if they were its
# choices. So the pilot appends an explicit format line and reads the numbers.
# This is a *pilot deviation* from the committed ask and is recorded per trial.
FORMAT_PICK_N = ("\n\nStart your reply with just the {n} numbers, comma-separated, "
                 "on a line of their own. Then give your reasons.")


# --------------------------------------------------------------------------- #
# task definitions
# --------------------------------------------------------------------------- #
def _entry(task: str) -> str:
    return os.path.join(TASKS, task, "__init__.py")


def load_task(task: str, poolname: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return pool(_entry(task), poolname)


def s9_options() -> List[Dict[str, Any]]:
    rows, _ = load_task("s9_neighborhood", "neighborhoods")
    return rows


def s12_options(topic: str) -> List[Dict[str, Any]]:
    rows, _ = load_task("s12_explain", "points")
    return [r for r in rows if r["topic"] == topic]


def s13_bugs() -> Dict[str, List[Dict[str, Any]]]:
    rows, _ = load_task("s13_patch", "patches")
    by_bug: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bug[r["bid"]].append(r)
    return dict(by_bug)


def render_option(task: str, row: Dict[str, Any]) -> str:
    if task == "s9_neighborhood":
        return f"{row['name']} — {row['description']}"
    if task == "s12_explain":
        return row["text"]
    if task == "s13_patch":
        return row["patch"]
    raise ValueError(task)


# --------------------------------------------------------------------------- #
# the order control
# --------------------------------------------------------------------------- #
def matched_orders(n_options: int, n_personas: int = ITEMS_PER_BUCKET) -> List[Tuple[List[int], str]]:
    """The same (rotation, direction) set for every bucket. See the module docstring.

    Rotation r of ``range(n)`` is ``[r, r+1, ..., n-1, 0, ..., r-1]``. With six
    personas we use rotations 0..5, each forward and reversed: twelve orders,
    identical in all three buckets, so position bias is a constant offset on every
    bucket mean.
    """
    out: List[Tuple[List[int], str]] = []
    for r in range(n_personas):
        base = [(r + i) % n_options for i in range(n_options)]
        out.append((base, "fwd"))
        out.append((list(reversed(base)), "rev"))
    return out


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
_NUM_LINE = re.compile(r"^[^0-9]{0,12}((?:\d{1,2})(?:\s*[,、和and]+\s*\d{1,2})*)", re.I)


def parse_numbers(text: str, n_expected: int, n_options: int) -> Optional[List[int]]:
    """The leading run of distinct 1..n_options integers, or None.

    Deliberately strict. `None` means "we could not read this", never a guess --
    a record that cannot be read is kept and excluded from the DV, which is the
    only way a parse failure stays visible instead of becoming a number.
    """
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NUM_LINE.match(line)
        if not m:
            continue
        nums = [int(x) for x in re.findall(r"\d{1,2}", m.group(1))]
        nums = [x for x in nums if 1 <= x <= n_options]
        seen, uniq = set(), []
        for x in nums:
            if x not in seen:
                seen.add(x); uniq.append(x)
        if len(uniq) >= n_expected:
            return uniq[:n_expected]
        return None
    return None


def dv_from_picks(shown: Sequence[Dict[str, Any]], picks: Sequence[int]) -> Dict[str, Any]:
    """Mean right_c over the picked options. Positive = the right-coded pole."""
    chosen = [shown[i - 1] for i in picks]
    return {"picked_ids": [c.get("nid") or c.get("pid") for c in chosen],
            "right_c_mean": statistics.fmean(float(c["right_c"]) for c in chosen)}


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def load_items() -> List[Dict[str, Any]]:
    per: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(ITEMS_FILE, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "explore":
                continue
            scores = row.get("image_scores") or []
            mean = sum(scores) / len(scores) if scores else 0.0
            row["bucket"] = "low" if mean < -0.5 else ("high" if mean > 0.5 else "mid")
            per[row["bucket"]].append(row)
    out: List[Dict[str, Any]] = []
    for b in BUCKETS:
        out.extend(per[b][:ITEMS_PER_BUCKET])
    return out


def _question(ask: str, shown: Sequence[Dict[str, Any]], task: str, n_picks: int) -> str:
    lines = [f"{i + 1}. {render_option(task, row)}" for i, row in enumerate(shown)]
    return ask + "\n\n" + "\n".join(lines) + FORMAT_PICK_N.format(n=n_picks)


def build_plan(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    specs: List[Tuple[str, str, str, List[Dict[str, Any]], int]] = [
        ("s9_neighborhood", "shortlist", "ask.txt", s9_options(), 3),
        ("s12_explain", "inflation", "ask_inflation.txt", s12_options("inflation"), 3),
        ("s12_explain", "sky_blue", "ask_sky_blue.txt", s12_options("sky_blue"), 3),
    ]
    for task, qid, askfile, options, n_picks in specs:
        ask = prompt_text(_entry(task), askfile)
        orders = matched_orders(len(options))
        for scheme in ("chat", "agentic"):
            for row in items:
                item = Item.from_dict(row)
                idx = [r["item_id"] for r in items if r["bucket"] == row["bucket"]].index(item.item_id)
                for order, arm in (orders[idx * 2], orders[idx * 2 + 1]):
                    shown = [options[i] for i in order]
                    question = _question(ask, shown, task, n_picks)
                    messages, tools = build_scheme_messages(scheme, item.image_paths, question)
                    plan.append({
                        "task": task, "qid": qid, "scheme": scheme, "order_arm": arm,
                        "bucket": row["bucket"], "item_id": item.item_id,
                        "image_mean": item.image_mean, "order": order,
                        "shown": shown, "n_picks": n_picks, "n_options": len(options),
                        "trial": Trial(
                            surface=task, item_id=item.item_id, condition="photos",
                            conversation=Conversation(messages=messages,
                                                      images=list(item.image_paths)),
                            candidates=[], probe_points=[], max_new_tokens=700,
                            variant={"scheme": scheme, "question": qid, "order_arm": arm},
                            meta={"family": "forced_choice", "scheme": scheme,
                                  "question": question, "tools": tools, "pilot": True},
                        )})
    # s13: the negative control. One bug per persona slot, same four patches.
    bugs = s13_bugs()
    ask_t = prompt_text(_entry("s13_patch"), "ask.j2") if os.path.exists(
        os.path.join(TASKS, "s13_patch", "prompts", "ask.j2")) else None
    for bid, patches in sorted(bugs.items()):
        orders = matched_orders(len(patches))
        symptom = patches[0]["symptom"]
        buggy = patches[0]["buggy_code"]
        ask = (f"This function is returning the wrong thing. {symptom}\n\n"
               f"```python\n{buggy}\n```\n\n"
               "Which one of these four changes fixes it? "
               "Answer with the number and nothing else.")
        for scheme in ("chat", "agentic"):
            for row in items:
                item = Item.from_dict(row)
                idx = [r["item_id"] for r in items if r["bucket"] == row["bucket"]].index(item.item_id)
                for order, arm in (orders[idx * 2], orders[idx * 2 + 1]):
                    shown = [patches[i] for i in order]
                    lines = [f"{i + 1}. {render_option('s13_patch', r)}"
                             for i, r in enumerate(shown)]
                    question = ask + "\n\n" + "\n".join(lines)
                    messages, tools = build_scheme_messages(scheme, item.image_paths, question)
                    plan.append({
                        "task": "s13_patch", "qid": bid, "scheme": scheme, "order_arm": arm,
                        "bucket": row["bucket"], "item_id": item.item_id,
                        "image_mean": item.image_mean, "order": order,
                        "shown": shown, "n_picks": 1, "n_options": len(patches),
                        "trial": Trial(
                            surface="s13_patch", item_id=item.item_id, condition="photos",
                            conversation=Conversation(messages=messages,
                                                      images=list(item.image_paths)),
                            candidates=[], probe_points=[], max_new_tokens=120,
                            variant={"scheme": scheme, "question": bid, "order_arm": arm},
                            meta={"family": "forced_choice", "scheme": scheme,
                                  "question": question, "tools": tools, "pilot": True},
                        )})
    return plan


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def phase_run(limit: int = 0) -> int:
    from bench.adaptors.vllm_server import VLLMServerAdaptor
    os.makedirs(OUT_DIR, exist_ok=True)
    adaptor = VLLMServerAdaptor(
        model=os.environ.get("VLLM_SERVED_MODEL", "qwen3-vl-8b-instruct"),
        base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8073/v1"),
        seed=SEED)
    adaptor.setup()

    plan = build_plan(load_items())
    if limit:
        plan = plan[:limit]
    print(f"[run] {len(plan)} trials", flush=True)

    n_err = 0
    with open(TRIALS_PATH, "a", encoding="utf-8") as handle:
        for i, e in enumerate(plan):
            started = time.time()
            try:
                resp = adaptor.run(e["trial"])
                text_out, err = resp.text, None
            except Exception as exc:                    # noqa: BLE001
                text_out, err = "", f"{type(exc).__name__}: {exc}"
                n_err += 1
            picks = parse_numbers(text_out, e["n_picks"], e["n_options"]) if text_out else None
            rec: Dict[str, Any] = {
                "task": e["task"], "qid": e["qid"], "scheme": e["scheme"],
                "order_arm": e["order_arm"], "bucket": e["bucket"],
                "item_id": e["item_id"], "image_mean": e["image_mean"],
                "order": e["order"], "picks": picks, "parse_ok": picks is not None,
                "text": text_out, "error": err, "ms": (time.time() - started) * 1000.0,
            }
            if picks:
                if e["task"] == "s13_patch":
                    chosen = e["shown"][picks[0] - 1]
                    rec.update({"correct": bool(chosen["correct"]),
                                "picked_ids": [chosen["pid"]]})
                else:
                    rec.update(dv_from_picks(e["shown"], picks))
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(plan)}] err={n_err}", flush=True)
    print(f"[run] done, errors={n_err} -> {TRIALS_PATH}", flush=True)
    return 0


def phase_report() -> int:
    rows = [json.loads(l) for l in open(TRIALS_PATH, encoding="utf-8") if l.strip()]
    print(f"\n=== {len(rows)} records ===")
    for task_qid in sorted({(r["task"], r["qid"]) for r in rows}):
        task, qid = task_qid
        sub = [r for r in rows if r["task"] == task and r["qid"] == qid]
        ok = [r for r in sub if r.get("parse_ok")]
        print(f"\n--- {task} / {qid}   n={len(sub)}  parse_ok={len(ok)}")
        if task == "s13_patch":
            for scheme in ("chat", "agentic"):
                acc = [r["correct"] for r in ok if r["scheme"] == scheme]
                by_b = {b: [r["correct"] for r in ok
                            if r["scheme"] == scheme and r["bucket"] == b] for b in BUCKETS}
                cells = "  ".join(
                    f"{b}={statistics.fmean(v):.2f}(n={len(v)})" for b, v in by_b.items() if v)
                if acc:
                    print(f"    {scheme:<8} accuracy {statistics.fmean(acc):.3f}   {cells}")
            continue
        for scheme in ("chat", "agentic"):
            by_b = {b: [r["right_c_mean"] for r in ok
                        if r["scheme"] == scheme and r["bucket"] == b] for b in BUCKETS}
            if not any(by_b.values()):
                continue
            cells = "  ".join(f"{b}={statistics.fmean(v):+.3f}(n={len(v)})"
                              for b, v in by_b.items() if v)
            span = (statistics.fmean(by_b["high"]) - statistics.fmean(by_b["low"])
                    if by_b["low"] and by_b["high"] else None)
            print(f"    {scheme:<8} {cells}   span={span:+.3f}" if span is not None
                  else f"    {scheme:<8} {cells}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["plan", "run", "report"], default="run")
    p.add_argument("--limit", type=int, default=0, help="cap trials; 0 = all")
    a = p.parse_args()
    if a.phase == "plan":
        plan = build_plan(load_items())
        print(f"{len(plan)} trials")
        seen = defaultdict(int)
        for e in plan:
            seen[(e["task"], e["qid"], e["scheme"])] += 1
        for k, v in sorted(seen.items()):
            print(f"  {k}: {v}")
        sys.exit(0)
    sys.exit(phase_run(a.limit) if a.phase == "run" else phase_report())


if __name__ == "__main__":
    main()
