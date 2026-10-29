"""face_swap v4 -- the best-token swap across the whole portrait, not only the face.

As v3, except that the cells that can move are every cell of the 600 x 800 frame (background,
hair, clothes and the grey outside the photo included), not only the cells that are person in
both photos; the top-ranked cells may now be background. Each trial records how many moved
cells were person in the base and in the donor.

The 20 people whose own tokens read most liberal and the 20 most conservative
(``bench_llm/datasets/face_swap/faces.json``), each photo aligned on the face so cells line
up (``build faces``), are paired most extreme with most extreme. In each pair one photo runs
(the base) and a share of its cells takes the other photo's tokens at the same positions
(``bench_llm/splice.py``, mode ``ranked``, region ``all``):

* ``c2l`` -- a liberal face takes the conservative face's most conservative tokens;
* ``l2c`` -- a conservative face takes the liberal face's most liberal tokens;
* shares 2, 5, 10, 25, 50 and 100% of all the cells;
* ``top``: the cells ranked by the donor's own image-token probe score, read in the same pass
  (combined_ideology_headwise_linear, top 16 heads, as the token scoring that picked the
  faces); ``random``: as many cells at random, the control;
* ``normal``: each of the 40 photos as it is.

Readings as v1 (both questions, both orders): the answer, the text-only probe at the last
prompt token, and the whole-word log-odds on the conservative-coded side. Each trial records
the cells moved and their scores.

The extremes are all women on the liberal side and all men on the conservative side: the
token probe tracks gender. The contrast that matters, top against random at the same share,
is within a pair, so gender is held fixed in it.

    python -m bench_llm.datasets.face_swap.build faces           # first, once
    python -m bench_llm.tasks.face_swap.v4.pilot plan
    python -m bench_llm.tasks.face_swap.v4.pilot run --model qwen3_vl
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

TASK = "face_swap/v4"
TITLE = "best-token swap across the whole portrait"
OUT_DIR = sources.repo_root() / "runs" / "bench_llm" / "face_swap" / "v4"
FACES = Path(__file__).resolve().parents[3] / "datasets" / "face_swap" / "faces.json"
DATA = "data/face_swap_v3"
ASPECT = 800 / 600
SHARES = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00)
HOWS = ("top", "random")
DIRECTIONS = {"c2l": ("liberal", "conservative", 1), "l2c": ("conservative", "liberal", -1)}
CONDITIONS = ("normal", "top", "random")
REGION = "all"        # every cell can move; v3 moved only cells that are person in both photos


def load_items(root: Optional[str] = None) -> List[Item]:
    sel = json.loads(FACES.read_text(encoding="utf-8"))
    frame = lambda u: sources.resolve(f"{DATA}/frames/{u}.jpg", root)  # noqa: E731
    mask = lambda u: sources.resolve(f"{DATA}/masks/{u}.png", root)    # noqa: E731
    items = []
    for side, tag in (("liberal", "lib"), ("conservative", "con")):
        for i, p in enumerate(sel[side]):
            items.append(Item(f"normal_{tag}_p{i:02d}", [frame(p["uuid"])], {
                "condition": "normal", "pair": i, "base": p["uuid"], "base_side": side,
                "base_score": round(p["fg_mean"], 4)}))
    n = min(len(sel["liberal"]), len(sel["conservative"]))
    for d, (base_side, donor_side, direction) in DIRECTIONS.items():
        for i in range(n):
            b, o = sel[base_side][i], sel[donor_side][i]
            for share in SHARES:
                for how in HOWS:
                    items.append(Item(f"{how}_{d}_p{i:02d}_s{round(share * 100):03d}", [frame(b["uuid"])], {
                        "condition": how, "direction": d, "pair": i, "share": share,
                        "base": b["uuid"], "base_side": base_side, "base_score": round(b["fg_mean"], 4),
                        "donor": o["uuid"], "donor_side": donor_side, "donor_score": round(o["fg_mean"], 4),
                        "splice": {"mode": "ranked", "donor_image": frame(o["uuid"]), "base_mask": mask(b["uuid"]),
                                   "donor_mask": mask(o["uuid"]), "aspect": ASPECT, "share": share, "how": how,
                                   "direction": direction, "region": REGION,
                                   "seed": 1000 * i + round(share * 100)}}))
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
    """Per model, direction, question, how and share: the mean shift from the base face's own
    run (probe and log-odds), averaged over pairs and both orders; + is toward conservative."""
    report: Dict[str, Any] = {}
    print(f"{TITLE}  [{TASK}]  {len(rows)} records")
    mean = lambda a: sum(a) / len(a) if a else float("nan")  # noqa: E731
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model and not r.get("error")]
        own = {}
        for r in mine:
            if r["item"]["condition"] == "normal":
                key = (r["item"]["base"], r["variant"]["question"], r["variant"]["order"])
                own[key] = (r["outcome"].get("value"), r["outcome"]["extra"].get("log_odds"))
        print(f"\n== {model}")
        print(f"  {'dir':<5}{'question':<9}{'how':<8}" + "".join(f"{round(s * 100):>6}%" for s in SHARES) + "   (probe shift)")
        for d in DIRECTIONS:
            for q in ("lean", "latte"):
                for how in HOWS:
                    cells = []
                    for share in SHARES:
                        dp, dl = [], []
                        for r in mine:
                            it = r["item"]
                            if it["condition"] != how or it["direction"] != d or it["share"] != share or r["variant"]["question"] != q:
                                continue
                            base = own.get((it["base"], q, r["variant"]["order"]))
                            lo = r["outcome"]["extra"].get("log_odds")
                            if base and base[0] is not None and r["outcome"].get("value") is not None:
                                dp.append(r["outcome"]["value"] - base[0])
                            if base and base[1] is not None and lo is not None:
                                dl.append(lo - base[1])
                        cells.append((mean(dp), mean(dl), len(dp)))
                        report.setdefault(model, {})[f"{d}/{q}/{how}/{share}"] = {"d_probe": mean(dp), "d_log_odds": mean(dl), "n": len(dp)}
                    print(f"  {d:<5}{q:<9}{how:<8}" + "".join(f"{c[0]:>+7.3f}" for c in cells))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=f"{TITLE} pilot")
    p.add_argument("phase", nargs="?", default="plan", choices=["plan", "run", "summary"])
    p.add_argument("--model", default="qwen3_vl", choices=sorted(v1.MODELS))
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-dir", default="results/probes")
    p.add_argument("--only", action="append", default=None, help="run only these item ids (repeatable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--runs", action="append", default=None)
    a = p.parse_args(argv)
    run_dir = Path(a.out) if a.out else OUT_DIR / a.model
    if a.phase == "plan":
        items = load_items()
        n = {c: sum(1 for it in items if it.data["condition"] == c) for c in CONDITIONS}
        print(f"{TITLE}  [{TASK}]  {len(items)} images {n} x {len(variants())} asks -> {len(items) * len(variants())} trials per model")
        return 0
    if a.phase == "run":
        spec = dict(v1.MODELS[a.model])
        if a.model_path:
            spec["model_path"] = a.model_path
        adaptor = ADAPTORS["hf_probe"](probe_prefix=v1.PROBE_PREFIX, probe_type=v1.PROBE_TYPE, top_k=v1.TOP_K,
                                       data_dir=a.data_dir, seed=a.seed, **spec)
        # normal runs first, then shares from small to large, so a partial run already has the curve's start
        items = sorted(load_items(), key=lambda it: (it.data["condition"] != "normal", it.data.get("share", 0)))
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
