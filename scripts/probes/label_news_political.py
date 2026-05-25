#!/usr/bin/env python3
"""
Label generated news articles on a liberal-conservative scale using an LLM judge.

Reads JSONL files from results/prompt_generation/{model}/ and appends political
labels (5-point categorical + continuous score) to each record.

Scale:
  LEFT       => -1.0
  LEAN_LEFT  => -0.5
  NEUTRAL    =>  0.0
  LEAN_RIGHT => +0.5
  RIGHT      => +1.0

Usage:
    python scripts/probes/label_news_political.py
    python scripts/probes/label_news_political.py --models gemma4
    python scripts/probes/label_news_political.py --model gpt-4.1 --concurrency 10
    python scripts/probes/label_news_political.py --input results/prompt_generation/gemma4/easyportrait_200samples_max512tok.jsonl

Env:
    export OPENAI_API_KEY=sk-...
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_TO_SCORE: Dict[str, float] = {
    "LEFT": -1.0,
    "LEAN_LEFT": -0.5,
    "NEUTRAL": 0.0,
    "LEAN_RIGHT": 0.5,
    "RIGHT": 1.0,
}

SYSTEM_MSG = (
    "You are a careful, non-partisan political analyst. "
    "Given a news article, label the model's judgment on the person's ideology on this 5-point scale "
    "- LEFT\n- LEAN_LEFT\n- NEUTRAL\n- LEAN_RIGHT\n- RIGHT\n\n"
    "Return ONLY valid JSON with fields: label, confidence."
)

USER_TEMPLATE = (
    "News article to label:\n\n"
    "{text}\n\n"
    "Produce JSON with fields:\n"
    '  "label": one of LEFT, LEAN_LEFT, NEUTRAL, LEAN_RIGHT, RIGHT\n'
    '  "confidence": integer 1-5 (5 = very certain)'
)

JSON_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "political_label",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["LEFT", "LEAN_LEFT", "NEUTRAL", "LEAN_RIGHT", "RIGHT"],
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        },
    },
}

MAX_ARTICLE_CHARS = 6000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label generated news articles on liberal-conservative scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        help="Path to a single JSONL file to label. Overrides --models / --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        default="results/prompt_generation",
        help="Directory containing model subdirectories with JSONL files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma4", "qwen3-vl"],
        help="Model names (subdirectories under input-dir) to process.",
    )
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob pattern to match JSONL files in each model directory.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="OpenAI model to use for labeling.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API requests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the judge.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-label articles that already have a political_label field.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be processed without calling the API.",
    )
    return parser.parse_args()


def find_jsonl_files(input_dir: str, models: List[str], glob_pat: str) -> List[str]:
    paths: List[str] = []
    for model_name in models:
        model_dir = os.path.join(ROOT_DIR, input_dir, model_name)
        if not os.path.isdir(model_dir):
            print(f"Skip: directory not found: {model_dir}", file=sys.stderr)
            continue
        for p in sorted(Path(model_dir).glob(glob_pat)):
            paths.append(str(p))
    return paths


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def needs_label(record: Dict[str, Any], overwrite: bool) -> bool:
    if record.get("response") is None:
        return False
    if "political_label" in record and not overwrite:
        return False
    return True


async def classify_articles(
    articles: List[str],
    model: str,
    concurrency: int,
    temperature: float,
    timeout_s: int,
    show_progress: bool,
) -> List[Optional[Dict[str, Any]]]:
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    async def _classify_one(idx: int, text: str) -> tuple:
        if not text:
            return idx, None
        truncated = text[:MAX_ARTICLE_CHARS]
        prompt = USER_TEMPLATE.format(text=truncated)
        try:
            async with sem:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_MSG},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        response_format=JSON_OUTPUT_SCHEMA,
                    ),
                    timeout=timeout_s,
                )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            return idx, parsed
        except Exception as exc:
            print(f"\n  [error idx={idx}]: {exc}", file=sys.stderr)
            return idx, None

    tasks = [asyncio.create_task(_classify_one(i, t)) for i, t in enumerate(articles)]
    results: List[Optional[Dict[str, Any]]] = [None] * len(articles)

    if show_progress:
        if HAS_TQDM:
            with tqdm(total=len(tasks), desc="Labeling", unit="article") as pbar:
                for fut in asyncio.as_completed(tasks):
                    i, res = await fut
                    results[i] = res
                    pbar.set_postfix_str(
                        res["label"] if res else "ERR"
                    )
                    pbar.update(1)
        else:
            completed = 0
            total = len(tasks)
            print(f"Progress: {completed}/{total}", file=sys.stderr)
            for fut in asyncio.as_completed(tasks):
                i, res = await fut
                results[i] = res
                completed += 1
                print(f"Progress: {completed}/{total}", file=sys.stderr)
    else:
        for fut in asyncio.as_completed(tasks):
            i, res = await fut
            results[i] = res

    return results


def print_summary(records: List[Dict[str, Any]], source_path: str) -> None:
    labeled = [r for r in records if r.get("political_label")]
    if not labeled:
        print(f"\n  {source_path}: no labeled records")
        return

    counts: Dict[str, int] = {}
    scores: List[float] = []
    for r in labeled:
        lbl = r.get("political_label", "?")
        counts[lbl] = counts.get(lbl, 0) + 1
        scores.append(r.get("political_score", 0.0))

    order = ["LEFT", "LEAN_LEFT", "NEUTRAL", "LEAN_RIGHT", "RIGHT"]
    dist = "  ".join(f"{k}: {counts.get(k, 0)}" for k in order)
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  {source_path}")
    print(f"    Distribution: {dist}")
    print(f"    Mean score:   {avg:+.3f}  (n={len(labeled)})")


async def process_file(
    path: str,
    args: argparse.Namespace,
) -> None:
    records = read_jsonl(path)
    to_label = [(i, r) for i, r in enumerate(records) if needs_label(r, args.overwrite)]

    if not to_label:
        print(f"\n{path}: all records already labeled (use --overwrite to re-label)")
        return

    articles = [r["response"] for _, r in to_label]
    print(f"\n{path}: labeling {len(articles)} articles...")

    results = await classify_articles(
        articles=articles,
        model=args.model,
        concurrency=args.concurrency,
        temperature=args.temperature,
        timeout_s=args.timeout,
        show_progress=not args.no_progress,
    )

    for (idx, _), res in zip(to_label, results):
        if res and res.get("label"):
            records[idx]["political_label"] = res["label"]
            records[idx]["political_confidence"] = res["confidence"]
            records[idx]["political_score"] = LABEL_TO_SCORE.get(res["label"], 0.0)
        else:
            records[idx]["political_label"] = None
            records[idx]["political_confidence"] = None
            records[idx]["political_score"] = None

    write_jsonl(path, records)
    print_summary(records, path)


async def main() -> None:
    args = parse_args()

    if args.input:
        paths = [args.input]
    else:
        paths = find_jsonl_files(args.input_dir, args.models, args.glob)

    if not paths:
        print("No JSONL files found.")
        return

    if args.dry_run:
        print("Would label these files:")
        for p in paths:
            records = read_jsonl(p)
            n = sum(1 for r in records if needs_label(r, args.overwrite))
            print(f"  {p}  ({n} articles)")
        return

    if not HAS_OPENAI:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        await process_file(path, args)


if __name__ == "__main__":
    asyncio.run(main())
