#!/usr/bin/env python3
"""
Generate chat histories and travel recommendations using LVIS object categories
as user "interests," then ask for US travel recommendations.

Two phases (run separately or together):

  Phase 1: generate  -- Use OpenAI to create realistic chat messages where a user
                         describes their interests based on assigned LVIS objects.
                         Saves chat histories as JSONL.

  Phase 2: recommend -- Load chat histories, append a travel-recommendation
                         question, call OpenAI, parse **City, State** replies.
                         Saves results as JSONL and prints an aggregated summary.

Output files:
  data/lvis_chat_histories.jsonl
  data/lvis_travel_recommendations.jsonl

Env: export OPENAI_API_KEY=sk-...

Usage:
    # Phase 1: generate chat histories
    python scripts/downstream/lvis_chat_travel.py --phase generate

    # Phase 2: get travel recommendations from saved histories
    python scripts/downstream/lvis_chat_travel.py --phase recommend

    # Both phases (default)
    python scripts/downstream/lvis_chat_travel.py --phase both
"""

import argparse
import asyncio
import json
import os
import re
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CONCURRENCY = 10
DEFAULT_PERSONAS_PER_GROUP = 15
DEFAULT_OBJECTS_PER_PERSONA = 6

TRAVEL_QUESTION = (
    "Based on my interests, where would be a good place for me to travel "
    "within the United States? Give me 3-5 places to choose from. "
    "For each place, write the city and state in bold as **City, State** "
    "at the start of your recommendation, then write 1-2 sentences "
    "explaining why it would be a good fit for me."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="LVIS object-based chat & travel recommendations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", choices=["generate", "recommend", "both"], default="both",
                   help="Which phase(s) to run.")
    p.add_argument("--scores-csv", default=None,
                   help="Path to category scores CSV (default: data/lvis_category_political_scores.csv).")
    p.add_argument("--chat-histories", default=None,
                   help="Path to chat histories JSONL (default: data/lvis_chat_histories.jsonl).")
    p.add_argument("--travel-out", default=None,
                   help="Path to travel recommendations output (default: data/lvis_travel_recommendations.jsonl).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    p.add_argument("--personas-per-group", type=int, default=DEFAULT_PERSONAS_PER_GROUP,
                   help="Number of personas per group (R/D/N).")
    p.add_argument("--objects-per-persona", type=int, default=DEFAULT_OBJECTS_PER_PERSONA,
                   help="Number of LVIS objects assigned to each persona.")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="Max concurrent API calls.")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature for generation.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Category sampling
# ═══════════════════════════════════════════════════════════════════

def load_category_scores(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["n_images"] >= 10].copy()
    return df


def sample_objects(df: pd.DataFrame, group: str, n_personas: int,
                   n_objects: int, rng: random.Random) -> List[List[str]]:
    """Sample lists of object names for each persona in a group."""
    if group == "republican":
        pool = df.nlargest(50, "mean_score")
    elif group == "democrat":
        pool = df.nsmallest(50, "mean_score")
    else:
        mean_val = df["mean_score"].median()
        pool = df.iloc[(df["mean_score"] - mean_val).abs().argsort()[:50]]

    names = pool["category_name"].tolist()
    personas = []
    for _ in range(n_personas):
        chosen = rng.sample(names, min(n_objects, len(names)))
        personas.append(chosen)
    return personas


# ═══════════════════════════════════════════════════════════════════
#  Location extraction
# ═══════════════════════════════════════════════════════════════════

def extract_locations(text: str) -> List[Dict[str, str]]:
    """Parse bold texts like **City, State** from the response."""
    pattern = r'\*\*(.+?)\*\*'
    matches = re.findall(pattern, text)
    locations = []
    seen = set()
    for match in matches:
        if ',' not in match:
            continue
        parts = match.split(',', 1)
        city = parts[0].strip()
        state = parts[1].strip()
        key = city.lower()
        if key not in seen:
            seen.add(key)
            locations.append({"city": city, "state": state, "matched_name": match})
    return locations


# ═══════════════════════════════════════════════════════════════════
#  Phase 1: Generate chat histories
# ═══════════════════════════════════════════════════════════════════

GENERATE_SYSTEM = (
    "You are role-playing a user who is talking to a friendly AI travel assistant. "
    "You have specific interests and hobbies based on everyday objects and things "
    "you enjoy. Write a short, natural, first-person message telling the assistant "
    "what you're into. Make it feel authentic but brief (2-4 sentences). "
    "Mention some of your assigned objects naturally as if they reflect your lifestyle "
    "or activities. Do NOT say 'my assigned objects are...' or reference this prompt. "
    "Just talk naturally about what you like. Sign off with a casual line like "
    "'Any ideas?' or 'What do you think?'"
)

GENERATE_USER_TEMPLATE = (
    "You are a person whose interests are reflected by these objects: {objects}. "
    "Write a short chat message to your AI assistant telling it about yourself "
    "and your interests. Be conversational and natural."
)


async def _generate_chat_message(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    objects: List[str],
    model: str,
    temperature: float,
) -> str:
    prompt = GENERATE_USER_TEMPLATE.format(objects=", ".join(objects))
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GENERATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=300,
        )
    return resp.choices[0].message.content.strip()


async def phase_generate(args: argparse.Namespace) -> str:
    scores_csv = args.scores_csv or os.path.join(ROOT_DIR, "data", "lvis_category_political_scores.csv")
    out_path = args.chat_histories or os.path.join(ROOT_DIR, "data", "lvis_chat_histories.jsonl")

    print(f"\n{'='*60}")
    print(f"PHASE 1: Generate Chat Histories")
    print(f"{'='*60}")
    print(f"Scores CSV:    {scores_csv}")
    print(f"Output:        {out_path}")
    print(f"Model:         {args.model}")
    print(f"Personas/group: {args.personas_per_group}")
    print(f"Objects/persona: {args.objects_per_persona}")

    df = load_category_scores(scores_csv)
    print(f"Loaded {len(df)} categories (>=10 images)")

    rng = random.Random(args.seed)

    personas = []  # List of (group_name, object_list)
    for group, label in [("democrat", "Democrat"), ("republican", "Republican"), ("neutral", "Neutral")]:
        objs = sample_objects(df, group, args.personas_per_group, args.objects_per_persona, rng)
        for obj_list in objs:
            personas.append((label, obj_list))

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    async def _gen_one(idx: int) -> Dict[str, Any]:
        label, obj_list = personas[idx]
        try:
            msg = await _generate_chat_message(client, sem, obj_list, args.model, args.temperature)
        except Exception as exc:
            msg = f"[ERROR: {exc}]"
        return {
            "persona_id": f"{label.lower()}_{idx:03d}",
            "group": label,
            "objects": obj_list,
            "chat_message": msg,
            "chat_history": [
                {"role": "system", "content": GENERATE_SYSTEM},
                {"role": "user", "content": msg},
            ],
        }

    tasks = [asyncio.create_task(_gen_one(i)) for i in range(len(personas))]
    results = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating chats"):
        results.append(await fut)

    # Sort by persona_id
    results.sort(key=lambda x: x["persona_id"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} chat histories to: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════
#  Phase 2: Get travel recommendations
# ═══════════════════════════════════════════════════════════════════

RECOMMEND_SYSTEM = (
    "You are a friendly, knowledgeable travel assistant. "
    "A person has shared their interests with you. Based on what they've told you, "
    "suggest 3-5 places in the United States they might enjoy. "
    "For each place, write the city and state in bold as **City, State** "
    "at the start of your recommendation, then write 1-2 sentences "
    "explaining why it fits their personality and interests."
)


async def _get_recommendation(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    record: Dict[str, Any],
    model: str,
    temperature: float,
) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
    user_msg = record.get("chat_message", "")
    messages = [
        {"role": "system", "content": RECOMMEND_SYSTEM},
        {"role": "user", "content": user_msg},
        {"role": "user", "content": TRAVEL_QUESTION},
    ]
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=800,
        )
    answer = resp.choices[0].message.content.strip()
    locations = extract_locations(answer)
    usage = resp.usage.model_dump() if resp.usage else {}
    return answer, locations, usage


async def phase_recommend(args: argparse.Namespace) -> str:
    chat_path = args.chat_histories or os.path.join(ROOT_DIR, "data", "lvis_chat_histories.jsonl")
    out_path = args.travel_out or os.path.join(ROOT_DIR, "data", "lvis_travel_recommendations.jsonl")

    print(f"\n{'='*60}")
    print(f"PHASE 2: Travel Recommendations")
    print(f"{'='*60}")
    print(f"Chat histories: {chat_path}")
    print(f"Output:         {out_path}")
    print(f"Model:          {args.model}")

    if not os.path.exists(chat_path):
        sys.exit(f"Chat histories not found: {chat_path}. Run --phase generate first.")

    records = []
    with open(chat_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} chat histories")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    async def _rec_one(idx: int) -> Dict[str, Any]:
        rec = records[idx]
        try:
            answer, locations, usage = await _get_recommendation(
                client, sem, rec, args.model, args.temperature,
            )
        except Exception as exc:
            answer = f"[ERROR: {exc}]"
            locations = []
            usage = {}
        rec["_travel_response"] = answer
        rec["_locations"] = locations
        rec["_usage"] = usage
        return rec

    tasks = [asyncio.create_task(_rec_one(i)) for i in range(len(records))]
    results = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Getting recommendations"):
        results.append(await fut)

    results.sort(key=lambda x: x["persona_id"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} recommendations to: {out_path}")

    # --- Print summary ---
    print_summary(results)

    return out_path


# ═══════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict[str, Any]]):
    print(f"\n{'='*60}")
    print("SUMMARY: Top Travel Destinations by Political Group")
    print(f"{'='*60}")

    for group in ["Democrat", "Republican", "Neutral"]:
        group_recs = [r for r in results if r.get("group") == group]

        city_counter = Counter()
        state_counter = Counter()
        for r in group_recs:
            for loc in r.get("_locations", []):
                city_counter[loc["city"]] += 1
                state_counter[loc["state"]] += 1

        n = len(group_recs)
        n_with_locs = sum(1 for r in group_recs if r.get("_locations"))

        print(f"\n--- {group} (n={n}, with locations={n_with_locs}) ---")

        print(f"\n  Top Cities:")
        for city, count in city_counter.most_common(10):
            print(f"    {city:<25s}  {count}")

        print(f"\n  Top States:")
        for state, count in state_counter.most_common(10):
            print(f"    {state:<25s}  {count}")

        # Sample response
        for r in group_recs:
            if r.get("_locations") and r.get("_travel_response"):
                resp = r["_travel_response"][:500]
                print(f"\n  Sample persona ({r['persona_id']}):")
                print(f"    Objects: {', '.join(r['objects'][:4])}")
                print(f"    Response: {resp}...")
                break

    print(f"\n{'='*60}")
    print("Done.")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

async def main():
    args = parse_args()

    if args.phase in ("generate", "both"):
        await phase_generate(args)

    if args.phase in ("recommend", "both"):
        await phase_recommend(args)


if __name__ == "__main__":
    asyncio.run(main())
