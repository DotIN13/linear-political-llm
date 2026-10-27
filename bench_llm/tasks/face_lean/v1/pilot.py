"""face_lean v1 -- liberal or conservative, in one word, and what the probe reads first.

Twenty faces (``faces.csv``): twelve members of Congress, three from each corner of
party by how Republican the combined-ideology probe read their portrait, and eight
EasyPortrait faces, the four read most Republican and the four read most Democratic.
Each image is the 800-pixel copy the per-patch maps were scored on.

One question (``ask.j2``), in one word order, image first. Each record keeps two
readings of one prompt:

* ``score`` -- the text-only probe (``textual_ideology_headwise_linear``, its 16 best
  heads) at the last prompt token, the position whose output is the answer's first
  token; positive reads conservative, negative liberal. This is the outcome value.
* ``word`` -- the one word the model wrote, greedy, and ``label``: liberal,
  conservative or other.

Needs the ``hf_probe`` adaptor, so it runs on a GPU node in the repo's probe
environment, one model per run:

    python -m bench_llm.tasks.face_lean.v1.pilot plan
    python -m bench_llm.tasks.face_lean.v1.pilot run --model qwen3_vl
    python -m bench_llm.tasks.face_lean.v1.pilot summary --runs runs/bench_llm/face_lean/v1/qwen3_vl
    python -m bench_llm.tasks.face_lean.v1.pilot export --runs DIR1 --runs DIR2 --to data.js
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
from bench_llm.types import Item, Outcome, Response, Trial

TASK = "face_lean/v1"
TITLE = "liberal or conservative, in one word"
HERE = Path(__file__).resolve().parent
OUT_DIR = sources.repo_root() / "runs" / "bench_llm" / "face_lean" / "v1"
FACES = HERE / "faces.csv"

MODELS: Dict[str, Dict[str, str]] = {
    "qwen3_vl": {"model": "qwen3-vl-8b-instruct", "model_family": "qwen3-vl",
                 "model_path": "/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"},
    "gemma4": {"model": "gemma-4-31b-it", "model_family": "gemma4",
               "model_path": "/project/jevans/tzhang3/models/gemma-4-31B-it"},
}
PROBE_PREFIX = "textual_ideology"
PROBE_TYPE = "headwise_linear"
TOP_K = 16
MAX_NEW_TOKENS = 8


def load_faces(root: Optional[str] = None) -> List[Item]:
    rows = sources.read_csv(FACES, numeric=("nominate_dim1",))
    return sources.to_items(rows, id_field="item_id", path_field="image_path", root=root,
                            data_fields=("set", "group", "name", "party", "nominate_dim1"))


# --- the ask -----------------------------------------------------------------
def variants() -> List[Dict[str, Any]]:
    return [{"order": "liberal_first", "probe": f"{PROBE_PREFIX}_{PROBE_TYPE}", "top_k": TOP_K}]


def build(item: Item, variant: Dict[str, Any]) -> Trial:
    text = prompts.render(HERE / "ask.j2")
    return Trial(item_id=item.item_id, conversation=prompts.conversation(text, item.image_paths),
                 variant=dict(variant), max_new_tokens=MAX_NEW_TOKENS, meta={"question": text})


# --- the read ----------------------------------------------------------------
WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")
LABELS = {"liberal": "liberal", "conservative": "conservative"}
REFUSAL = re.compile(r"i(?:'m| am) (?:sorry|not able|unable)|i can(?:no|')t|cannot (?:determine|tell|say)", re.I)


def first_word(text: str) -> Optional[str]:
    m = WORD.search(text or "")
    return m.group(0) if m else None


def read(resp: Response, trial: Trial) -> Outcome:
    text = (resp.text or "").strip()
    word = first_word(text)
    label = LABELS.get((word or "").lower().rstrip("-'"), "other")
    probe = resp.probe or {}
    score = probe.get("score")
    return Outcome(value=float(score) if score is not None else None, extra={
        "word": word, "label": label, "answer_text": text,
        "refusal": readers.matches(text, REFUSAL),
        "probe": probe.get("probe"), "top_k": probe.get("top_k"),
        "read_at": probe.get("last_prompt_token")})


# --- the summary -------------------------------------------------------------
def summarize(rows: List[Dict[str, Any]], title: str = TITLE, task: str = TASK) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    print(f"{title}  [{task}]  {len(rows)} records")
    for model in sorted({r["model"] for r in rows}):
        group = sorted((r for r in rows if r["model"] == model),
                       key=lambda r: (r["item"].get("set", ""), r["item"].get("group", ""), r["item_id"]))
        print(f"\n== {model}")
        print(f"{'face':<34}{'party':<12}{'word':<16}{'score':>8}")
        for r in group:
            it, ex = r.get("item") or {}, r["outcome"].get("extra") or {}
            v = r["outcome"].get("value")
            print(f"{(it.get('name') or r['item_id'])[:33]:<34}{(it.get('party') or '-'):<12}"
                  f"{str(ex.get('word') or '-')[:15]:<16}{(f'{v:+.3f}' if v is not None else '-'):>8}")
        by: Dict[str, List[float]] = {}
        for r in group:
            v = r["outcome"].get("value")
            if v is not None:
                by.setdefault(r["outcome"]["extra"]["label"], []).append(v)
        print("  mean last-token score by word:")
        for label, vals in sorted(by.items()):
            print(f"    {label:<14}{len(vals):>3}  {sum(vals) / len(vals):+.3f}")
        report[model] = {k: {"n": len(v), "mean_score": sum(v) / len(v)} for k, v in by.items()}
    return report


def export(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One entry per face, with each model's word and score, for the board."""
    faces: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        it = r.get("item") or {}
        f = faces.setdefault(r["item_id"], {
            "item_id": r["item_id"], "set": it.get("set"), "group": it.get("group"), "name": it.get("name"),
            "party": it.get("party") or None, "nominate": it.get("nominate_dim1"),
            "image": Path((r.get("messages") or [{}])[-1]["content"][0].get("image", "")).name, "models": {}})
        ex = r["outcome"].get("extra") or {}
        f["models"][r["model"]] = {"word": ex.get("word"), "label": ex.get("label"),
                                   "text": ex.get("answer_text"), "score": r["outcome"].get("value"),
                                   "read_at": ex.get("read_at"), "error": r.get("error")}
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
    p.add_argument("--to", default=None, help="export: where to write data.js")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.out) if args.out else OUT_DIR / args.model

    if args.phase == "plan":
        items = load_faces()
        cells = runner.cells(variants(), items)
        print(f"{TITLE}  [{TASK}]  {len(items)} faces x {len(variants())} variant -> {len(cells)} trials per model")
        for it in items:
            print(f"  {it.item_id:<22}{it.data.get('set'):<14}{it.data.get('name')}")
        print("  ask: | " + build(*cells[0][::-1]).meta["question"])
        return 0

    if args.phase == "run":
        spec = dict(MODELS[args.model])
        if args.model_path:
            spec["model_path"] = args.model_path
        adaptor = ADAPTORS["hf_probe"](probe_prefix=PROBE_PREFIX, probe_type=PROBE_TYPE, top_k=TOP_K,
                                       data_dir=args.data_dir, seed=args.seed, **spec)
        cells = runner.cells(variants(), load_faces())
        runner.run_cells(task=TASK, cells=cells, build=build, read=read, adaptor=adaptor,
                         out_dir=run_dir, workers=1, limit=args.limit,
                         extra={"faces_sha16": sources.file_sha16(FACES)})
        rows = runner.read_trials([run_dir])
        (run_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=1) + "\n", encoding="utf-8")
        return 0

    dirs = [Path(d) for d in (args.runs or [run_dir])]
    rows = runner.read_trials(dirs)
    if args.phase == "summary":
        summarize(rows)
        return 0
    data = export(rows)
    to = Path(args.to or dirs[0] / "data.js")
    to.write_text("window.FACE_LEAN=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                  encoding="utf-8")
    print(f"{len(data)} faces -> {to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
