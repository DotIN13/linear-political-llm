"""party_look v1 -- does this image look Democratic or Republican, on 1 to 7?

One image, one question, no persona and no memory. The model is shown a single
image from a source pool and asked to place it on a 1 to 7 scale from strongly
associated with one party to strongly associated with the other. Its answer is
set against what the image already carries:

* the probe score (``image_mean``: the combined-ideology probe on the image's own
  tokens, positive = conservative) -- does what the model says about an image
  follow what its activations hold about it?
* for ``congress``, the member's party and DW-NOMINATE score -- does it read a
  politician's side from a portrait? (The probe was trained partly on portraits,
  so here the party, not the probe, is the ground truth.)

The scale is asked in both orientations (``dem_low``: 1 = Democrats, and
``rep_low``: 1 = Republicans) so a preference for a digit cancels in the average.
Each record keeps three readings of one reply:

* ``answer`` -- the digit the model actually wrote, and its value;
* ``logprobs`` -- the first token's log-probability of each digit 1..7 (None when
  a digit was outside the 20 tokens returned), plus the top 20 as returned;
* ``value`` -- the probability-weighted value of the digits, the score, with
  ``log_odds`` of the Republican half over the Democratic half beside it.

Every value is on one axis, -1 Democratic .. +1 Republican.

    python -m bench_llm.tasks.party_look.v1.pilot plan --source congress
    python -m bench_llm.tasks.party_look.v1.pilot run --source lvis --base-url URL --out DIR
    python -m bench_llm.tasks.party_look.v1.pilot summary --runs DIR1 --runs DIR2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_llm import prompts, readers, sources, stats
from bench_llm import run as runner
from bench_llm.adaptors import ADAPTORS
from bench_llm.types import Item, Outcome, Response, Trial

TASK = "party_look/v1"
TITLE = "does this image look Democratic or Republican"
HERE = Path(__file__).resolve().parent
OUT_DIR = sources.repo_root() / "runs" / "bench_llm" / "party_look" / "v1"

# --- the pools ---------------------------------------------------------------
SCORING = "results/token_scoring/qwen3_vl"
PROBE = "combined_ideology_headwise_linear"
SOURCES: Dict[str, str] = {
    name: f"{SCORING}/{name}/prompt_token_stats_{PROBE}.csv"
    for name in ("lvis", "unsplash", "easyportrait", "congress")
}
LEGISLATORS_CSV = "data/legislators_116_119.csv"
NOMINATE_CSV = "data/HS116_members.csv"


def usable(raw: Dict[str, str]) -> bool:
    """Integrity only, as in bench_v2's sampler: the image was fully tokenized."""
    return int(raw.get("image_token_mismatch") or 0) == 0 and int(raw.get("num_image_tokens") or 0) > 0


def load_source(name: str, strata: int, per_stratum: int, seed: int,
                root: Optional[str] = None) -> Tuple[List[Item], Dict[str, Any]]:
    path = sources.resolve(SOURCES[name], root)
    rows = sources.dedupe(sources.read_csv(path, numeric=("image_mean",), keep=usable), "record_id")
    sources.stratify(rows, "image_mean", strata)
    picked = sources.sample_per_stratum(rows, per_stratum, seed, order_by="record_id")
    for r in picked:
        r["source"] = name
    if name == "congress":
        parties = sources.read_table(sources.resolve(LEGISLATORS_CSV, root), key="bioguide",
                                     latest_by="congress")
        nominate = sources.read_table(sources.resolve(NOMINATE_CSV, root), key="bioguide_id",
                                      numeric=("nominate_dim1",))
        for r in picked:
            bid = r["record_id"].split("_", 1)[1]
            r.update(bioguide=bid, party=(parties.get(bid) or {}).get("party"),
                     nominate_dim1=(nominate.get(bid) or {}).get("nominate_dim1"))
    items = sources.to_items(picked, id_field="record_id", path_field="image_path", root=root,
                             data_fields=("source", "record_name", "image_mean", "stratum",
                                          "bioguide", "party", "nominate_dim1"))
    header = {"source": name, "csv": SOURCES[name], "csv_sha16": sources.file_sha16(path),
              "probe": PROBE, "n_pool": len(rows), "strata": strata,
              "per_stratum": per_stratum, "seed": seed, "n_items": len(items)}
    return items, header


# --- the ask -----------------------------------------------------------------
ORDERS = ("dem_low", "rep_low")
MAX_NEW_TOKENS = 8
TOP_LOGPROBS = 20
MIN_MASS = 0.5
DEM, REP = "Democrats", "Republicans"
SCALE = tuple(str(k) for k in range(1, 8))
DIGIT = re.compile(r"[1-7]")
REFUSAL = re.compile(r"i(?:'m| am) (?:sorry|not able|unable)|i can(?:no|')t (?:help|assist|provide|make)", re.I)


def variants() -> List[Dict[str, Any]]:
    return [{"format": "scale", "order": o} for o in ORDERS]


def dem_first(order: str) -> bool:
    return order == "dem_low"


def question(variant: Dict[str, Any]) -> str:
    low, high = (DEM, REP) if dem_first(variant["order"]) else (REP, DEM)
    return prompts.render(HERE / "ask_scale.j2", first=low, second=high, low=low, high=high)


def build(item: Item, variant: Dict[str, Any]) -> Trial:
    text = question(variant)
    return Trial(item_id=item.item_id, conversation=prompts.conversation(text, item.image_paths),
                 variant=dict(variant), max_new_tokens=MAX_NEW_TOKENS, top_logprobs=TOP_LOGPROBS,
                 meta={"question": text})


# --- the read ----------------------------------------------------------------
def digit_values(variant: Dict[str, Any]) -> Dict[str, float]:
    """Each digit on -1 Democratic .. +1 Republican."""
    sign = 1 if dem_first(variant["order"]) else -1
    return {k: sign * (int(k) - 4) / 3 for k in SCALE}


def digit_logprobs(logprobs: Optional[Dict[str, float]]) -> Dict[str, Optional[float]]:
    """Each digit's first-token log-probability; tokens that clean to one digit are summed."""
    out: Dict[str, Optional[float]] = {k: None for k in SCALE}
    for token, lp in (logprobs or {}).items():
        t = readers.clean_token(token)
        if t in out:
            out[t] = lp if out[t] is None else math.log(math.exp(out[t]) + math.exp(lp))
    return out


def read(resp: Response, trial: Trial) -> Outcome:
    text = (resp.text or "").strip()
    values = digit_values(trial.variant)
    written = DIGIT.search(text)
    probs, mass = readers.option_probs(resp.logprobs, SCALE)
    top, top_p = readers.top_option(probs)
    rep = [k for k, v in values.items() if v > 0]
    dem = [k for k, v in values.items() if v < 0]
    answer = written.group(0) if written else None
    extra = {
        "answer": answer,
        "answer_value": values[answer] if answer else None,
        "answer_text": text,
        "logprobs": digit_logprobs(resp.logprobs),
        "top_logprobs": dict(resp.logprobs or {}),
        "probs": probs, "mass": mass, "top": top, "top_p": top_p,
        "answer_is_top": bool(answer and answer == top),
        "log_odds": readers.log_odds(probs, rep, dem) if mass >= MIN_MASS else None,
        "refusal": readers.matches(text, REFUSAL),
    }
    return Outcome(value=readers.expected_value(probs, values, mass, MIN_MASS), extra=extra)


# --- the summary -------------------------------------------------------------
def per_image(rows: List[Dict[str, Any]], field: str = "value") -> Dict[str, Dict[str, Any]]:
    """One entry per image: a reading averaged over the two orientations, plus its data."""
    acc: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out = r.get("outcome") or {}
        v = out.get("value") if field == "value" else (out.get("extra") or {}).get(field)
        e = acc.setdefault(r["item_id"], {"data": r.get("item") or {}, "vals": []})
        if v is not None:
            e["vals"].append(float(v))
    return {k: {**e["data"], "score": sum(e["vals"]) / len(e["vals"])}
            for k, e in acc.items() if e["vals"]}


def fmt_r(r: Optional[float], n: int) -> str:
    if r is None:
        return "-"
    se = stats.corr_se(r, n)
    return f"{r:+.3f} (se {se:.3f}, n {n})" if se else f"{r:+.3f} (n {n})"


READINGS = (("value", "expected value"), ("answer_value", "written answer"), ("log_odds", "log-odds"))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[str((r.get("item") or {}).get("source"))].append(r)
    report: Dict[str, Any] = {}
    print(f"{TITLE}  [{TASK}]  {len(rows)} records")
    for source, group in sorted(by_source.items()):
        print(f"\n== {source}")
        print(f"{'order':<10}{'n':>6}{'read':>6}{'refused':>8}{'mean':>8}{'top>=.99':>9}{'written=top':>12}  written digits 1..7")
        for v in variants():
            cell = [r for r in group if r.get("variant") == v]
            if not cell:
                continue
            ex = [r["outcome"].get("extra") or {} for r in cell]
            vals = [r["outcome"]["value"] for r in cell if r["outcome"].get("value") is not None]
            mean = f"{sum(vals) / len(vals):+.3f}" if vals else "-"
            sure = sum(1 for e in ex if (e.get("top_p") or 0) >= 0.99) / len(cell)
            agree = sum(1 for e in ex if e.get("answer_is_top")) / len(cell)
            hist = " ".join(str(sum(1 for e in ex if e.get("answer") == k)) for k in SCALE)
            print(f"{v['order']:<10}{len(cell):>6}{len(vals):>6}{sum(1 for e in ex if e.get('refusal')):>8}"
                  f"{mean:>8}{sure:>9.0%}{agree:>12.0%}  {hist}")
        rep: Dict[str, Any] = {}
        print("  reading (mean of both orientations) against the image's probe score, Spearman:")
        for field, label in READINGS:
            img = per_image(group, field)
            pairs = [(e["image_mean"], e["score"]) for e in img.values() if e.get("image_mean") is not None]
            r = stats.spearman([p for p, _ in pairs], [s for _, s in pairs]) if pairs else None
            rep[f"{field}.spearman_probe"] = r
            print(f"    {label:<16}{fmt_r(r, len(pairs))}")
            if source == "congress":
                R = [e["score"] for e in img.values() if e.get("party") == "Republican"]
                D = [e["score"] for e in img.values() if e.get("party") == "Democrat"]
                a = stats.auc(R, D)
                nom = [(e["nominate_dim1"], e["score"]) for e in img.values() if e.get("nominate_dim1") is not None]
                rn = stats.spearman([x for x, _ in nom], [y for _, y in nom]) if nom else None
                rep[f"{field}.auc_party"] = a
                rep[f"{field}.spearman_nominate"] = rn
                print(f"      party AUC {a:.3f} (R {len(R)}, D {len(D)}; 0.5 = chance)" if a is not None else "      party AUC -")
                print(f"      vs DW-NOMINATE {fmt_r(rn, len(nom))}")
        report[source] = rep
    return report


# --- main --------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{TITLE} pilot")
    p.add_argument("phase", nargs="?", default="plan", choices=["plan", "run", "summary", "all"])
    p.add_argument("--source", action="append", default=None, help=f"repeatable; {sorted(SOURCES)}")
    p.add_argument("--strata", type=int, default=10)
    p.add_argument("--per-stratum", type=int, default=50, help="images per probe-score stratum; 0 = all")
    p.add_argument("--adaptor", default="openai_compat", choices=sorted(ADAPTORS))
    p.add_argument("--model", default="qwen3-vl-8b-instruct")
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="run only the first N cells")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--runs", action="append", default=None, help="summary: repeatable run dirs")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.out) if args.out else OUT_DIR

    if args.phase in {"plan", "run", "all"}:
        items, headers = [], []
        for name in args.source or sorted(SOURCES):
            got, header = load_source(name, args.strata, args.per_stratum, args.seed)
            items += got
            headers.append(header)
        cells = runner.cells(variants(), items)

    if args.phase == "plan":
        print(f"{TITLE}  [{TASK}]")
        for h in headers:
            print(f"  {h['source']:<13}{h['n_items']:>6} of {h['n_pool']} images "
                  f"({h['strata']} strata x {h['per_stratum'] or 'all'}, csv {h['csv_sha16']})")
        print(f"  x {len(variants())} variants -> {len(cells)} trials")
        variant, item = cells[0]
        print(f"  example: {item.item_id} {variant}")
        for line in build(item, variant).meta["question"].splitlines():
            print(f"    | {line}")
        return 0

    if args.phase in {"run", "all"}:
        adaptor = ADAPTORS[args.adaptor](model=args.model, base_url=args.base_url, seed=args.seed)
        runner.run_cells(task=TASK, cells=cells, build=build, read=read, adaptor=adaptor,
                         out_dir=run_dir, workers=args.workers, limit=args.limit,
                         extra={"sources": headers})

    if args.phase in {"summary", "all"}:
        dirs = [Path(d) for d in (args.runs or [run_dir])]
        report = summarize(runner.read_trials(dirs))
        out = dirs[0] / "summary.json"
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"\nsummary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
