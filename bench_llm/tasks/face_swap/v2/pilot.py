"""face_swap v2 -- the background moved by token splicing, against v2's pixel composites.

Person P2 (the 10 people of ``bench_llm/datasets/face_swap``) and background photo P1 (its
10 background photos), all framed to 600 x 800 so each model gives them the same token grid.
Four conditions, each asked face_swap v1's four questions (politics and latte, both orders):

* ``swap``   -- P2's photo runs, but its background cells take P1's background tokens; cells
  where P1's own person sat take the mean of P1's background tokens (``bench_llm/splice.py``);
* ``fill``   -- P2's photo, with exactly those cells set to the mean of P2's own background
  tokens: what the filling does by itself;
* ``normal`` -- P2's photo as it is;
* ``pixel``  -- v2's pixel composite of the same pair (``data/face_swap_v2/images``).

Readings as v1: the answer, the text-only probe at the last prompt token, and the whole-word
log-odds on the conservative-coded side. Each spliced trial also records the grid and how many
tokens came from where.

    python -m bench_llm.datasets.face_swap.build frames          # first, once
    python -m bench_llm.tasks.face_swap.v2.pilot plan
    python -m bench_llm.tasks.face_swap.v2.pilot run --model qwen3_vl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_llm import run as runner
from bench_llm import sources
from bench_llm.adaptors import ADAPTORS
from bench_llm.tasks.face_swap.v1 import pilot as v1
from bench_llm.types import Item, Trial

TASK = "face_swap/v2"
TITLE = "token-spliced backgrounds against pixel composites"
OUT_DIR = sources.repo_root() / "runs" / "bench_llm" / "face_swap" / "v2"
SELECTION = Path(__file__).resolve().parents[3] / "datasets" / "face_swap" / "selection.json"
DATA = "data/face_swap_v2"
ASPECT = 800 / 600
CONDITIONS = ("swap", "fill", "normal", "pixel")


def load_items(root: Optional[str] = None, conditions=CONDITIONS) -> List[Item]:
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    people = [dict(p, side=s) for s in ("liberal", "conservative") for p in sel["people"][s]]
    bgs = [dict(b, side=s) for s in ("liberal", "conservative") for b in sel["backgrounds"][s]]
    frame = lambda u: sources.resolve(f"{DATA}/frames/{u}.jpg", root)  # noqa: E731
    mask = lambda u: sources.resolve(f"{DATA}/masks/{u}.png", root)    # noqa: E731
    items = []
    for i, p in enumerate(people):
        base = {"person": p["uuid"], "person_side": p["side"], "person_score": round(p["fg_mean"], 4)}
        if "normal" in conditions:
            items.append(Item(f"normal_p{i:02d}", [frame(p["uuid"])], {**base, "condition": "normal",
                              "background": p["uuid"], "background_side": "own"}))
        for j, b in enumerate(bgs):
            pair = {**base, "background": b["uuid"], "background_side": b["side"],
                    "background_score": round(b["bg_mean"], 4)}
            for c in ("swap", "fill"):
                if c in conditions:
                    items.append(Item(f"{c}_p{i:02d}_b{j:02d}", [frame(p["uuid"])], {
                        **pair, "condition": c,
                        "splice": {"mode": "swap" if c == "swap" else "fill_check", "donor_image": frame(b["uuid"]),
                                   "base_mask": mask(p["uuid"]), "donor_mask": mask(b["uuid"]), "aspect": ASPECT}}))
            if "pixel" in conditions:
                items.append(Item(f"pixel_p{i:02d}_b{j:02d}",
                                  [sources.resolve(f"{DATA}/images/p{i:02d}_b{j:02d}.jpg", root)], {**pair, "condition": "pixel"}))
    return items


def variants() -> List[Dict[str, Any]]:
    return v1.variants()


def build(item: Item, variant: Dict[str, Any]) -> Trial:
    t = v1.build(item, variant)
    meta = dict(t.meta)
    if item.data.get("splice"):
        meta["splice"] = item.data["splice"]
    return Trial(item_id=t.item_id, conversation=t.conversation, variant=t.variant,
                 max_new_tokens=t.max_new_tokens, meta=meta)


read = v1.read


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per model, condition, question and order: the background effect per person
    (conservative-read minus liberal-read background), averaged, for probe and log-odds."""
    report: Dict[str, Any] = {}
    print(f"{TITLE}  [{TASK}]  {len(rows)} records")
    mean = lambda a: sum(a) / len(a) if a else float("nan")  # noqa: E731
    for model in sorted({r["model"] for r in rows}):
        print(f"\n== {model}")
        print(f"  {'condition':<9}{'question':<8}{'order':<20}{'d probe':>9}{'d log-odds':>12}{'people up':>11}")
        for cond in CONDITIONS:
            for v in variants():
                g = [r for r in rows if r["model"] == model and r["item"].get("condition") == cond
                     and r["variant"]["question"] == v["question"] and r["variant"]["order"] == v["order"]]
                per: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
                for r in g:
                    per.setdefault(r["item"]["person"], {}).setdefault(r["item"]["background_side"], []).append(r)
                dp, dl = [], []
                for d in per.values():
                    if "liberal" in d and "conservative" in d:
                        val = lambda side, f: mean([f(r) for r in d[side] if f(r) is not None])  # noqa: E731
                        s = lambda r: r["outcome"].get("value")  # noqa: E731
                        lo = lambda r: r["outcome"]["extra"].get("log_odds")  # noqa: E731
                        dp.append(val("conservative", s) - val("liberal", s))
                        dl.append(val("conservative", lo) - val("liberal", lo))
                if not dp:
                    continue
                up = sum(1 for x in dp if x > 0)
                print(f"  {cond:<9}{v['question']:<8}{v['order']:<20}{mean(dp):>+9.3f}{mean(dl):>+12.2f}{up:>8}/{len(dp)}")
                report.setdefault(model, {})[f"{cond}/{v['question']}/{v['order']}"] = {
                    "d_probe": mean(dp), "d_log_odds": mean(dl), "people_up": up, "n": len(dp)}
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=f"{TITLE} pilot")
    p.add_argument("phase", nargs="?", default="plan", choices=["plan", "run", "summary"])
    p.add_argument("--model", default="qwen3_vl", choices=sorted(v1.MODELS))
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-dir", default="results/probes")
    p.add_argument("--condition", action="append", default=None, choices=CONDITIONS)
    p.add_argument("--only", action="append", default=None, help="run only these item ids (repeatable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--runs", action="append", default=None)
    a = p.parse_args(argv)
    run_dir = Path(a.out) if a.out else OUT_DIR / a.model
    conds = tuple(a.condition or CONDITIONS)
    if a.phase == "plan":
        items = load_items(conditions=conds)
        n = {c: sum(1 for it in items if it.data["condition"] == c) for c in conds}
        print(f"{TITLE}  [{TASK}]  {len(items)} images {n} x {len(variants())} asks -> {len(items) * len(variants())} trials per model")
        return 0
    if a.phase == "run":
        spec = dict(v1.MODELS[a.model])
        if a.model_path:
            spec["model_path"] = a.model_path
        adaptor = ADAPTORS["hf_probe"](probe_prefix=v1.PROBE_PREFIX, probe_type=v1.PROBE_TYPE, top_k=v1.TOP_K,
                                       data_dir=a.data_dir, seed=a.seed, **spec)
        # the pixel and normal conditions first: they need no splicing and anchor the comparison
        items = sorted(load_items(conditions=conds), key=lambda it: CONDITIONS[::-1].index(it.data["condition"]))
        if a.only:
            items = [it for it in items if it.item_id in set(a.only)]
        runner.run_cells(task=TASK, cells=runner.cells(variants(), items), build=build, read=read,
                         adaptor=adaptor, out_dir=run_dir, workers=1, limit=a.limit)
        rows = runner.read_trials([run_dir])
        (run_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=1) + "\n", encoding="utf-8")
        return 0
    summarize(runner.read_trials([Path(d) for d in (a.runs or [run_dir])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
