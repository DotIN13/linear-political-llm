#!/usr/bin/env python3
"""
Label rewritten political text for bias on a 5-scale using OpenAI gpt-5 (async),
with structured output parsing via Pydantic.

Scale:
- LEFT
- LEAN_LEFT
- NEUTRAL
- LEAN_RIGHT
- RIGHT

Usage:
  python label_bias_async.py --input data.csv --output labeled.csv

Env:
  export OPENAI_API_KEY=sk-...
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from typing import List, Dict, Any, Optional, Literal

# pip install --upgrade openai pydantic tqdm
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, validator

# ---------- Optional progress ----------
try:
    from tqdm import tqdm  # pip install tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ---------- Parsing helpers ----------

ASSISTANT_BLOCK_REGEXES = [
    # Fallback: single quoted inner block
    re.compile(r'ASSISTANT:.*?([\s\S]*?)(?:USER|$)', re.IGNORECASE),
]

def extract_rewritten_text(answer_field: str) -> str:
    """
    The CSV's 'answer' cell contains both USER and ASSISTANT snippets.
    We pull the ASSISTANT rewritten quote block.
    """
    if not answer_field:
        return ""
    for rx in ASSISTANT_BLOCK_REGEXES:
        m = rx.search(answer_field)
        if m and m.group(1).strip():
            return m.group(1).strip()
    # If regex fails, as a last resort return the whole field
    return answer_field.strip()

# ---------- Structured Output Model ----------

LabelLiteral = Literal["LEFT", "LEAN_LEFT", "NEUTRAL", "LEAN_RIGHT", "RIGHT"]

class BiasLabel(BaseModel):
    label: LabelLiteral
    confidence: int = Field(ge=1, le=5)

# ---------- Prompting ----------

SYSTEM_MSG = (
    "You are a careful, non-partisan rater. "
    "Given a short political statement, label its ideological tilt on this 5-point scale in the 2019 US political context:\n"
    "- LEFT\n- LEAN_LEFT\n- NEUTRAL\n- LEAN_RIGHT\n- RIGHT\n\n"
    "Consider wording, framing, and policy valence. "
    "Return ONLY the structured object requested by the tool (no prose)."
)

USER_TEMPLATE = (
    "Statement to label:\n\n"
    "{text}\n\n"
    "Produce fields:\n"
    "  label: LEFT, LEAN_LEFT, NEUTRAL, LEAN_RIGHT, RIGHT\n"
    "  confidence: 1-5 (5 = very certain)"
)

# ---------- OpenAI call (async, structured parse) ----------

async def classify_texts_async(
    texts: List[str],
    model: str = "gpt-5",
    concurrency: int = 5,
    temperature: float = 0.0,
    timeout_s: int = 90,
    show_progress: bool = True,
) -> List[Optional[BiasLabel]]:
    """
    Run async classifications with a concurrency cap and progress updates.
    Returns list of BiasLabel or None on failure.
    """
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    async def _classify_one(idx: int, text: str) -> (int, Optional[BiasLabel]):
        if not text:
            print("ERROR: No text")
            return idx, None
        prompt = USER_TEMPLATE.format(text=text)
        try:
            async with sem:
                # Structured output parsing with Pydantic model
                resp = await asyncio.wait_for(
                    client.responses.parse(
                        model=model,
                        input=[
                            {"role": "system", "content": SYSTEM_MSG},
                            {"role": "user", "content": prompt},
                        ],
                        text_format=BiasLabel,
                    ),
                    timeout=timeout_s,
                )
            return idx, resp.output_parsed  # This is a BiasLabel instance
        except Exception as e:
            print(e)
            return idx, None

    tasks = [asyncio.create_task(_classify_one(i, t)) for i, t in enumerate(texts)]
    results: List[Optional[BiasLabel]] = [None] * len(texts)

    if show_progress:
        if _HAS_TQDM:
            with tqdm(total=len(tasks), desc="Labeling", unit="item") as pbar:
                for fut in asyncio.as_completed(tasks):
                    i, res = await fut
                    results[i] = res
                    pbar.update(1)
        else:
            completed = 0
            total = len(tasks)
            print(f"Progress: {completed}/{total}", file=sys.stderr, flush=True)
            for fut in asyncio.as_completed(tasks):
                i, res = await fut
                results[i] = res
                completed += 1
                print(f"Progress: {completed}/{total}", file=sys.stderr, flush=True)
    else:
        for fut in asyncio.as_completed(tasks):
            i, res = await fut
            results[i] = res

    return results

# ---------- CSV IO ----------

def read_rows(input_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def write_rows(output_path: str, rows: List[Dict[str, Any]], field_order: Optional[List[str]] = None) -> None:
    if not rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["parsed_label"])
        return
    headers = list(rows[0].keys())
    if "parsed_label" not in headers:
        headers.append("parsed_label")
    if "parsed_confidence" not in headers:
        headers.append("parsed_confidence")
    if field_order:
        ordered = [h for h in field_order if h in headers] + [h for h in headers if h not in field_order]
        headers = ordered
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------- Main ----------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Async bias labeling with gpt-5 (structured output).")
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--output", required=True, help="Path to write labeled CSV")
    p.add_argument("--model", default="gpt-5", help="Model name (default: gpt-5)")
    p.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--timeout", type=int, default=90, help="Per-request timeout (seconds)")
    p.add_argument("--no-progress", action="store_true", help="Disable progress display")
    return p

async def run(args):
    rows = read_rows(args.input)

    # Extract texts from 'answer' column (which includes the rewritten ASSISTANT block)
    texts: List[str] = []
    for row in rows:
        answer_field = row.get("answer", "") or ""
        rewritten = extract_rewritten_text(answer_field)
        texts.append(rewritten)

    # Classify asynchronously with progress
    results = await classify_texts_async(
        texts=texts,
        model=args.model,
        concurrency=args.concurrency,
        temperature=args.temperature,
        timeout_s=args.timeout,
        show_progress=not args.no_progress,
    )

    # Attach results back to rows
    for row, res in zip(rows, results):
        if isinstance(res, BiasLabel):
            row["parsed_label"] = res.label
            row["parsed_confidence"] = res.confidence
        else:
            row["parsed_label"] = ""
            row["parsed_confidence"] = ""

    # Preserve original column order, appending parsed_label/parsed_confidence at end
    original_order = list(rows[0].keys()) if rows else None
    write_rows(args.output, rows, field_order=original_order)

def main():
    args = build_argparser().parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
