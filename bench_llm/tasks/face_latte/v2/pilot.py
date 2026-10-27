"""face_latte v2 -- latte or coffee, in both orders, in one word, and what the probe reads first.

v1 asked "latte or coffee" only; v2 asks it both ways ("latte or coffee" and "coffee or
latte") so a pull toward the first-named drink shows up and can be averaged out. The
same 20 faces and images as face_lean v1 (its ``faces.csv``), in the user's voice
(``ask.j2``), image first. Each record
keeps two readings of one prompt:

* ``score`` -- the text-only political probe (``textual_ideology_headwise_linear``, its
  16 best heads) at the last prompt token, the position face_lean read too; positive
  reads conservative. This is the outcome value.
* ``word`` -- the one word the model wrote, greedy, and ``label``: latte, coffee or other.

``export`` joins each face with its face_lean record, so the board shows the political
word and the latte word side by side.

    python -m bench_llm.tasks.face_latte.v2.pilot plan
    python -m bench_llm.tasks.face_latte.v2.pilot run --model qwen3_vl
    python -m bench_llm.tasks.face_latte.v2.pilot summary --runs runs/bench_llm/face_latte/v2/qwen3_vl
    python -m bench_llm.tasks.face_latte.v2.pilot export --runs DIR --lean-runs DIR --to data.js
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_llm import prompts, readers, sources
from bench_llm import run as runner
from bench_llm.adaptors import ADAPTORS
from bench_llm.tasks.face_lean.v1 import pilot as lean
from bench_llm.types import Item, Outcome, Response, Trial

TASK = "face_latte/v2"
TITLE = "latte or coffee, both orders, in one word"
HERE = Path(__file__).resolve().parent
OUT_DIR = sources.repo_root() / "runs" / "bench_llm" / "face_latte" / "v2"
MODELS = lean.MODELS
PROBE_PREFIX, PROBE_TYPE, TOP_K = lean.PROBE_PREFIX, lean.PROBE_TYPE, lean.TOP_K
MAX_NEW_TOKENS = 8
load_faces = lean.load_faces


# --- the ask -----------------------------------------------------------------
ORDERS = {"latte_first": ("latte", "coffee"), "coffee_first": ("coffee", "latte")}


def variants() -> List[Dict[str, Any]]:
    return [{"order": o, "probe": f"{PROBE_PREFIX}_{PROBE_TYPE}", "top_k": TOP_K} for o in ORDERS]


def build(item: Item, variant: Dict[str, Any]) -> Trial:
    first, second = ORDERS[variant["order"]]
    text = prompts.render(HERE / "ask.j2", first=first, second=second)
    return Trial(item_id=item.item_id, conversation=prompts.conversation(text, item.image_paths),
                 variant=dict(variant), max_new_tokens=MAX_NEW_TOKENS, meta={"question": text})


# --- the read ----------------------------------------------------------------
LABELS = {"latte": "latte", "lattes": "latte", "coffee": "coffee"}


def read(resp: Response, trial: Trial) -> Outcome:
    text = (resp.text or "").strip()
    word = lean.first_word(text)
    label = LABELS.get((word or "").lower().rstrip("-'"), "other")
    probe = resp.probe or {}
    score = probe.get("score")
    return Outcome(value=float(score) if score is not None else None, extra={
        "word": word, "label": label, "answer_text": text,
        "refusal": readers.matches(text, lean.REFUSAL),
        "probe": probe.get("probe"), "top_k": probe.get("top_k"),
        "read_at": probe.get("last_prompt_token")})


# --- summary and export ------------------------------------------------------
def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    print(f"{TITLE}  [{TASK}]  {len(rows)} records")
    for model in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == model]
        print(f"\n== {model}")
        rep: Dict[str, Any] = {}
        for order in ORDERS:
            cell = [r for r in group if (r.get("variant") or {}).get("order") == order]
            counts: Dict[str, int] = {}
            for r in cell:
                lab = (r["outcome"].get("extra") or {}).get("label", "other")
                counts[lab] = counts.get(lab, 0) + 1
            vals = [r["outcome"]["value"] for r in cell if r["outcome"].get("value") is not None]
            mean = sum(vals) / len(vals) if vals else None
            print(f"  {order:<14}{len(cell):>3} trials  {counts}  mean probe {mean:+.3f}" if mean is not None
                  else f"  {order:<14}{len(cell):>3} trials  {counts}")
            rep[order] = {"n": len(cell), "labels": counts, "mean_score": mean}
        by_face: Dict[str, Dict[str, str]] = {}
        for r in group:
            by_face.setdefault(r["item_id"], {})[(r.get("variant") or {}).get("order")] = \
                (r["outcome"].get("extra") or {}).get("label")
        same = sum(1 for v in by_face.values() if len(set(v.values())) == 1 and len(v) == len(ORDERS))
        print(f"  same answer in both orders: {same} of {len(by_face)} faces")
        rep["same_both_orders"] = same
        report[model] = rep
    return report


def export(rows: List[Dict[str, Any]], lean_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """face_lean's export with each model's latte word and score, per order, beside the lean ones."""
    faces = {f["item_id"]: f for f in lean.export(lean_rows)} if lean_rows else {}
    for order in ORDERS:
        part = [r for r in rows if (r.get("variant") or {}).get("order") == order]
        for f in lean.export(part):
            base = faces.setdefault(f["item_id"], {**f, "models": {}})
            for model, m in f["models"].items():
                base["models"].setdefault(model, {}).setdefault("latte", {})[order] = m
    order = {it.item_id: i for i, it in enumerate(load_faces())}
    return sorted(faces.values(), key=lambda f: order.get(f["item_id"], 1 << 30))


# --- main --------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{TITLE} pilot")
    p.add_argument("phase", nargs="?", default="plan", choices=["plan", "run", "summary", "export"])
    p.add_argument("--model", default="qwen3_vl", choices=sorted(MODELS))
    p.add_argument("--model-path", default=None, help="override the model directory")
    p.add_argument("--data-dir", default="results/probes", help="where the trained probes are")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--runs", action="append", default=None, help="summary/export: repeatable run dirs")
    p.add_argument("--lean-runs", action="append", default=None, help="export: face_lean run dirs to join")
    p.add_argument("--to", default=None, help="export: where to write data.js")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.out) if args.out else OUT_DIR / args.model

    if args.phase == "plan":
        items = load_faces()
        cells = runner.cells(variants(), items)
        print(f"{TITLE}  [{TASK}]  {len(items)} faces x {len(variants())} variant -> {len(cells)} trials per model")
        variant, item = cells[0]
        print("  ask: | " + build(item, variant).meta["question"])
        return 0

    if args.phase == "run":
        spec = dict(MODELS[args.model])
        if args.model_path:
            spec["model_path"] = args.model_path
        adaptor = ADAPTORS["hf_probe"](probe_prefix=PROBE_PREFIX, probe_type=PROBE_TYPE, top_k=TOP_K,
                                       data_dir=args.data_dir, seed=args.seed, **spec)
        runner.run_cells(task=TASK, cells=runner.cells(variants(), load_faces()), build=build, read=read,
                         adaptor=adaptor, out_dir=run_dir, workers=1, limit=args.limit,
                         extra={"faces_sha16": sources.file_sha16(lean.FACES)})
        rows = runner.read_trials([run_dir])
        (run_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=1) + "\n", encoding="utf-8")
        return 0

    dirs = [Path(d) for d in (args.runs or [run_dir])]
    rows = runner.read_trials(dirs)
    if args.phase == "summary":
        summarize(rows)
        return 0
    lean_rows = runner.read_trials([Path(d) for d in args.lean_runs]) if args.lean_runs else []
    data = export(rows, lean_rows)
    to = Path(args.to or dirs[0] / "data.js")
    to.write_text("window.FACE_LATTE=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                  encoding="utf-8")
    print(f"{len(data)} faces -> {to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
