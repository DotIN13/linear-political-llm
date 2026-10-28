"""face_swap v1 -- the face_lean and face_latte questions on the background-swap composites.

The 110 images of ``bench_llm/datasets/face_swap`` (10 people on 10 backgrounds, plus
each person on their own edited background) are each asked two questions, each in both
word orders, image first:

* politics, face_lean's wording: "... more likely to be liberal leaning or conservative
  leaning? Answer with one word." and the same with conservative first;
* latte, face_latte's wording: "Take a guess, do you think I like latte or coffee? Answer
  with one word." and the same with coffee first.

Every trial keeps three readings of one prompt:

* ``word`` / ``label`` -- the answer, greedy;
* ``score`` -- the text-only probe (textual_ideology_headwise_linear, top 16 heads) at the last
  prompt token, + conservative; the outcome value;
* ``log_odds`` -- whole-word log-odds of the conservative-coded option over the other, spellings
  pooled: conservative - liberal for politics, coffee - latte for latte.

    python -m bench_llm.tasks.face_swap.v1.pilot plan
    python -m bench_llm.tasks.face_swap.v1.pilot run --model qwen3_vl
    python -m bench_llm.tasks.face_swap.v1.pilot summary --runs runs/bench_llm/face_swap/v1/qwen3_vl
"""

from __future__ import annotations

import argparse
import json
import math
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

TASK = "face_swap/v1"
TITLE = "politics and latte on background-swap composites"
HERE = Path(__file__).resolve().parent
OUT_DIR = sources.repo_root() / "runs" / "bench_llm" / "face_swap" / "v1"
MANIFEST = "data/face_swap/manifest.csv"
MODELS = lean.MODELS
PROBE_PREFIX, PROBE_TYPE, TOP_K = lean.PROBE_PREFIX, lean.PROBE_TYPE, lean.TOP_K
MAX_NEW_TOKENS = 8

# per question: the template, the conservative-coded option first, then the other
QUESTIONS = {
    "lean": {"ask": "ask_lean.j2", "con": "conservative", "other": "liberal",
             "spell": {"conservative": ("conservative", "Conservative"), "liberal": ("liberal", "Liberal")}},
    "latte": {"ask": "ask_latte.j2", "con": "coffee", "other": "latte",
              "spell": {"coffee": ("coffee", "Coffee"), "latte": ("latte", "Latte")}},
}


def load_items(root: Optional[str] = None) -> List[Item]:
    rows = sources.read_csv(sources.resolve(MANIFEST, root), numeric=("person_score", "background_score"))
    return sources.to_items(rows, id_field="item_id", path_field="image_path", root=root,
                            data_fields=("kind", "person", "person_side", "person_score",
                                         "background", "background_side", "background_score"))


# --- the ask -----------------------------------------------------------------
def variants() -> List[Dict[str, Any]]:
    out = []
    for q, spec in QUESTIONS.items():
        for first in (spec["other"], spec["con"]):   # the face_lean / face_latte order first
            out.append({"question": q, "order": f"{first}_first", "probe": f"{PROBE_PREFIX}_{PROBE_TYPE}",
                        "top_k": TOP_K})
    return out


def build(item: Item, variant: Dict[str, Any]) -> Trial:
    spec = QUESTIONS[variant["question"]]
    first = variant["order"].removesuffix("_first")
    second = spec["con"] if first == spec["other"] else spec["other"]
    text = prompts.render(HERE / spec["ask"], first=first, second=second)
    words = [w for ws in spec["spell"].values() for w in ws]
    return Trial(item_id=item.item_id, conversation=prompts.conversation(text, item.image_paths),
                 variant=dict(variant), max_new_tokens=MAX_NEW_TOKENS,
                 meta={"question": text, "candidates": words})


# --- the read ----------------------------------------------------------------
def logsumexp(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def odds(spec: Dict[str, Any], cands: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    lp = {opt: logsumexp([(cands.get(w) or {}).get("logprob") for w in ws]) for opt, ws in spec["spell"].items()}
    c, o = lp[spec["con"]], lp[spec["other"]]
    if c is None or o is None:
        return {"log_odds": None, "mass": None, f"logp_{spec['con']}": c, f"logp_{spec['other']}": o}
    return {"log_odds": c - o, "mass": math.exp(c) + math.exp(o), f"logp_{spec['con']}": c, f"logp_{spec['other']}": o}


def read(resp: Response, trial: Trial) -> Outcome:
    spec = QUESTIONS[trial.variant["question"]]
    text = (resp.text or "").strip()
    word = lean.first_word(text)
    w = (word or "").lower().rstrip("-'")
    label = w if w in spec["spell"] else "other"
    probe = resp.probe or {}
    score = probe.get("score")
    return Outcome(value=float(score) if score is not None else None, extra={
        "word": word, "label": label, "answer_text": text, **odds(spec, probe.get("candidates") or {}),
        "refusal": readers.matches(text, lean.REFUSAL), "probe": probe.get("probe"),
        "top_k": probe.get("top_k"), "read_at": probe.get("last_prompt_token")})


# --- the summary -------------------------------------------------------------
def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per model, question and order: the four person x background cells (swaps only), and the
    background effect per person (conservative-read minus liberal-read backgrounds)."""
    report: Dict[str, Any] = {}
    print(f"{TITLE}  [{TASK}]  {len(rows)} records")
    mean = lambda a: sum(a) / len(a) if a else float("nan")  # noqa: E731
    for model in sorted({r["model"] for r in rows}):
        print(f"\n== {model}")
        for v in variants():
            cell = [r for r in rows if r["model"] == model and r.get("variant", {}).get("question") == v["question"]
                    and r["variant"].get("order") == v["order"] and (r.get("item") or {}).get("kind") == "swap"]
            if not cell:
                continue
            con = QUESTIONS[v["question"]]["con"]
            print(f"  {v['question']:<6}{v['order']:<20}{'cell':<26}{'n':>3}{'probe':>8}{'log-odds':>10}{'% ' + con:>16}")
            for ps in ("liberal", "conservative"):
                for bs in ("liberal", "conservative"):
                    g = [r for r in cell if r["item"]["person_side"] == ps and r["item"]["background_side"] == bs]
                    sc = [r["outcome"]["value"] for r in g if r["outcome"].get("value") is not None]
                    lo = [r["outcome"]["extra"].get("log_odds") for r in g if r["outcome"]["extra"].get("log_odds") is not None]
                    pc = sum(1 for r in g if r["outcome"]["extra"].get("label") == con) / max(1, len(g))
                    print(f"  {'':<26}{ps + ' on ' + bs:<26}{len(g):>3}{mean(sc):>+8.3f}{mean(lo):>+10.2f}{pc:>15.0%}")
                    report.setdefault(model, {}).setdefault(f"{v['question']}/{v['order']}", {})[f"{ps}|{bs}"] = {
                        "n": len(g), "probe": mean(sc), "log_odds": mean(lo), "share_con": pc}
    return report


# --- main --------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=f"{TITLE} pilot")
    p.add_argument("phase", nargs="?", default="plan", choices=["plan", "run", "summary"])
    p.add_argument("--model", default="qwen3_vl", choices=sorted(MODELS))
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-dir", default="results/probes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--runs", action="append", default=None)
    a = p.parse_args(argv)
    run_dir = Path(a.out) if a.out else OUT_DIR / a.model
    if a.phase == "plan":
        items = load_items()
        cells = runner.cells(variants(), items)
        print(f"{TITLE}  [{TASK}]  {len(items)} images x {len(variants())} variants -> {len(cells)} trials per model")
        for v in variants():
            print(f"  {v['question']:<6}{v['order']:<20}| " + build(items[0], v).meta["question"])
        return 0
    if a.phase == "run":
        spec = dict(MODELS[a.model])
        if a.model_path:
            spec["model_path"] = a.model_path
        adaptor = ADAPTORS["hf_probe"](probe_prefix=PROBE_PREFIX, probe_type=PROBE_TYPE, top_k=TOP_K,
                                       data_dir=a.data_dir, seed=a.seed, **spec)
        runner.run_cells(task=TASK, cells=runner.cells(variants(), load_items()), build=build, read=read,
                         adaptor=adaptor, out_dir=run_dir, workers=1, limit=a.limit,
                         extra={"manifest_sha16": sources.file_sha16(sources.resolve(MANIFEST))})
        rows = runner.read_trials([run_dir])
        (run_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=1) + "\n", encoding="utf-8")
        return 0
    summarize(runner.read_trials([Path(d) for d in (a.runs or [run_dir])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
