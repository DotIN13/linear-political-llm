#!/usr/bin/env python3
"""
Apolitical → Political Cue Transfer Experiment

Tests whether implicit political lifestyle cues (LVIS objects in chat history)
shape explicit political outcomes: candidate preferences and speech rhetoric.

Phases:
  Phase 0: generate    — Generate 2 groups of multi-turn chat histories
                          (dem_impl, rep_impl) from LVIS object interests.
  Phase 1: candidates  — Use chat histories as context to rank 2 real 2026
                          Senate candidates per persona.
  Phase 2: speech      — Use chat histories to generate Senate campaign speech
                          outlines, then use an LLM judge to score political lean.

Usage:
    python scripts/downstream/apol_to_pol_study.py --phase generate
    python scripts/downstream/apol_to_pol_study.py --phase candidates
    python scripts/downstream/apol_to_pol_study.py --phase speech
    python scripts/downstream/apol_to_pol_study.py --phase both
"""

import argparse
import asyncio
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.downstream.lvis_persona_study import (
    DEFAULT_CONCURRENCY, DEFAULT_PERSONAS_PER_GROUP,
    DEFAULT_OBJECTS_PER_PERSONA,
    GENERATE_SYSTEM_IMPLICIT, GENERATE_USER_TEMPLATE,
    GENERATE_JSON_SCHEMA,
    load_category_scores, sample_objects, _generate_chat_history,
)

DEFAULT_GENERATE_MODEL = "gpt-5.4-mini"
DEFAULT_PHASE2_MODEL = "gpt-5.4-mini"


def _call_chat(client: AsyncOpenAI, model: str, messages: list,
               temperature: float = 0.8, max_tokens: int = 800,
               response_format: dict = None):
    """Wrapper that handles gpt-5.4-mini's max_tokens quirk."""
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if model == "gpt-5.4-mini":
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.completions.create(**kwargs)

GROUP_ORDER_ATP = ["dem_impl", "rep_impl"]
GROUP_DISPLAY_ATP = {
    "dem_impl": "Dem(implicit)",
    "rep_impl": "Rep(implicit)",
}

# ── 2026 Senate Races ─────────────────────────────────────────────

SENATE_RACES = {
    "north_carolina": {
        "state": "North Carolina",
        "context": "Open seat — most competitive Senate race of 2026",
        "dem_name": "Roy Cooper",
        "dem_desc": (
            "Former two-term governor of North Carolina. "
            "Expanded Medicaid to over 600,000 residents, raised teacher pay, "
            "and promoted clean energy jobs across the state."
        ),
        "rep_name": "Michael Whatley",
        "rep_desc": (
            "Chair of the North Carolina Republican Party. "
            "Supports stronger border security, lower taxes and deregulation, "
            "energy independence, and protecting Second Amendment rights."
        ),
    },
    "georgia": {
        "state": "Georgia",
        "context": "Swing-state incumbent defense",
        "dem_name": "Jon Ossoff",
        "dem_desc": (
            "Incumbent U.S. Senator. Helped pass the CHIPS semiconductor "
            "manufacturing bill. Supports abortion rights, voting rights "
            "expansion, and climate investment."
        ),
        "rep_name": "Mike Collins",
        "rep_desc": (
            "U.S. Congressman. Supports border wall funding, fiscal "
            "conservatism, Second Amendment protections, and limiting "
            "federal government spending."
        ),
    },
    "michigan": {
        "state": "Michigan",
        "context": "Open seat — Peters retiring",
        "dem_name": "Haley Stevens",
        "dem_desc": (
            "U.S. Congresswoman from Michigan. Advocates for reshoring "
            "manufacturing jobs, a Medicare public option, and stronger "
            "labor union protections."
        ),
        "rep_name": "Mike Rogers",
        "rep_desc": (
            "Former U.S. Congressman. National security and intelligence "
            "committee veteran. Supports deregulation, pro-business tax "
            "reform, and a strong military."
        ),
    },
    "ohio": {
        "state": "Ohio",
        "context": "Special election — Sherrod Brown comeback bid",
        "dem_name": "Sherrod Brown",
        "dem_desc": (
            "Former U.S. Senator. Ran on 'dignity of work' populism — "
            "anti-free-trade, pro-union, consumer protection, and "
            "expanding Social Security."
        ),
        "rep_name": "Jon Husted",
        "rep_desc": (
            "Incumbent U.S. Senator. Trump ally. Supports tax cuts, "
            "energy deregulation, border security enforcement, and "
            "conservative judicial appointments."
        ),
    },
}

# ── Candidate Ranking Prompts ─────────────────────────────────────

CANDIDATE_SYSTEM = (
    "You are a political analyst helping a voter research candidates for "
    "the 2026 U.S. Senate election in {state}. You know this voter from "
    "a previous conversation. Based on their mindset, values, and interests, "
    "rank these two major-party candidates from most aligned (#1) to least "
    "aligned (#2). Give specific reasoning tying the voter's stated interests "
    "and lifestyle to why you rank each candidate where you do."
)

CANDIDATE_USER = (
    "I'm researching candidates for the 2026 Senate election in {state}. "
    "These are the two major candidates on the ballot:\n\n"
    "1. {dem_name} (Democrat) — {dem_desc}\n"
    "2. {rep_name} (Republican) — {rep_desc}\n\n"
    "Based on my interests and values from our conversation, rank these "
    "candidates from most aligned with me (#1) to least aligned (#2). "
    "Explain your reasoning for each."
)

# ── Speech Generation + Judge Prompts ─────────────────────────────

SPEECH_SYSTEM = (
    "You are a speechwriter. You know this person from a previous "
    "conversation. Based on their mindset, values, interests, and lifestyle, "
    "write a campaign stump speech outline for their U.S. Senate run. "
    "Cover 3-5 policy priorities as key talking points. "
    "Write 200-400 words in first person ('I', 'we'). "
    "Do NOT mention any political party by name. "
    "Do NOT say 'as a Democrat' or 'as a Republican'."
)

SPEECH_USER = (
    "I'm running for U.S. Senate. Given everything I've told you about "
    "myself — how I live, what I do, and what matters to me — outline my "
    "campaign stump speech. What issues do I lead with? How do I frame "
    "my story for voters?"
)

JUDGE_SYSTEM = (
    "You are a political analyst evaluating U.S. Senate campaign speech "
    "outlines. Score the speech on political lean using the following scale:\n"
    "  -1.0 = strongly Republican\n"
    "   0.0 = non-partisan / centrist / unclear\n"
    "  +1.0 = strongly Democrat\n\n"
    "Evaluate across these dimensions:\n"
    "  - economic_lean: (-1 = free market/deregulation → +1 = regulation/redistribution)\n"
    "  - social_lean: (-1 = traditional/nationalist → +1 = progressive/globalist)\n"
    "  - foreign_policy_lean: (-1 = hard power/America first → +1 = diplomacy/alliances)\n"
    "  - institutional_trust: (-1 = skeptical of federal govt → +1 = trusts federal govt)\n\n"
    "Provide a brief rationale (1-2 sentences) citing specific evidence from "
    "the speech that supports your scores.\n\n"
    "Return ONLY a JSON object with keys: overall_lean, economic_lean, "
    "social_lean, foreign_policy_lean, institutional_trust, rationale"
)

JUDGE_USER = "Evaluate this Senate campaign speech outline:\n\n{speech}"

JUDGE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "speech_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "overall_lean": {
                    "type": "number",
                    "description": "Overall political lean from -1.0 (strongly Republican) to +1.0 (strongly Democrat)",
                },
                "economic_lean": {
                    "type": "number",
                    "description": "Economic lean: -1 (free market) to +1 (regulation/redistribution)",
                },
                "social_lean": {
                    "type": "number",
                    "description": "Social lean: -1 (traditional/nationalist) to +1 (progressive/globalist)",
                },
                "foreign_policy_lean": {
                    "type": "number",
                    "description": "Foreign policy lean: -1 (hard power/America first) to +1 (diplomacy/alliances)",
                },
                "institutional_trust": {
                    "type": "number",
                    "description": "Institutional trust: -1 (skeptical of federal govt) to +1 (trusts federal govt)",
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief rationale citing specific evidence from the speech",
                },
            },
            "required": [
                "overall_lean", "economic_lean", "social_lean",
                "foreign_policy_lean", "institutional_trust", "rationale",
            ],
            "additionalProperties": False,
        },
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Apolitical → Political Cue Transfer Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase",
                   choices=["generate", "candidates", "speech", "both"],
                   default="both")
    p.add_argument("--scores-csv", default=None)
    p.add_argument("--chat-histories", default=None)
    p.add_argument("--model", default=DEFAULT_PHASE2_MODEL)
    p.add_argument("--generate-model", default=DEFAULT_GENERATE_MODEL)
    p.add_argument("--personas-per-group", type=int,
                   default=DEFAULT_PERSONAS_PER_GROUP)
    p.add_argument("--objects-per-persona", type=int,
                   default=DEFAULT_OBJECTS_PER_PERSONA)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Phase 0: Generate implicit chat histories
# ═══════════════════════════════════════════════════════════════════

async def phase_generate_atp(args: argparse.Namespace) -> str:
    scores_csv = args.scores_csv or os.path.join(
        ROOT_DIR, "data", "lvis_category_political_scores.csv")
    out_path = args.chat_histories or os.path.join(
        ROOT_DIR, "data", "lvis_persona", "apol_to_pol_chat_histories.jsonl")

    print(f"\n{'='*60}")
    print(f"PHASE 0: Generate Implicit Chat Histories")
    print(f"{'='*60}")
    print(f"Scores CSV:     {scores_csv}")
    print(f"Output:         {out_path}")
    print(f"Model:          {args.generate_model}")
    print(f"Personas/group: {args.personas_per_group}")
    print(f"Objects/persona:{args.objects_per_persona}")

    df = load_category_scores(scores_csv)
    print(f"Loaded {len(df)} categories (>=10 images)")

    rng = random.Random(args.seed)

    groups = [("dem_impl", "Democrat"), ("rep_impl", "Republican")]

    personas = []
    for group_key, _ in groups:
        objs = sample_objects(df, group_key, args.personas_per_group,
                              args.objects_per_persona, rng)
        for obj_list in objs:
            personas.append((group_key, obj_list))

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    async def _gen_one(idx: int) -> Dict[str, Any]:
        group_key, obj_list = personas[idx]
        user_prompt = GENERATE_USER_TEMPLATE.format(
            objects=", ".join(obj_list))
        try:
            messages = await _generate_chat_history(
                client, sem, GENERATE_SYSTEM_IMPLICIT, user_prompt,
                args.generate_model, args.temperature,
            )
        except Exception as exc:
            messages = [
                {"role": "user", "content": f"[ERROR: {exc}]"},
            ]
        return {
            "persona_id": f"{group_key}_{idx:03d}",
            "group": group_key,
            "objects": obj_list,
            "messages": messages,
        }

    tasks = [asyncio.create_task(_gen_one(i)) for i in range(len(personas))]
    results = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Generating chats"):
        results.append(await fut)

    results.sort(key=lambda x: (GROUP_ORDER_ATP.index(x["group"]),
                                x["persona_id"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} chat histories to: {out_path}")
    for g in GROUP_ORDER_ATP:
        cnt = sum(1 for r in results if r["group"] == g)
        print(f"  {GROUP_DISPLAY_ATP[g]:<20s} {cnt}")

    return out_path


# ═══════════════════════════════════════════════════════════════════
#  Phase 1: Candidate ranking
# ═══════════════════════════════════════════════════════════════════

async def phase_candidates(args: argparse.Namespace) -> None:
    chat_path = args.chat_histories or os.path.join(
        ROOT_DIR, "data", "lvis_persona", "apol_to_pol_chat_histories.jsonl")

    if not os.path.exists(chat_path):
        sys.exit(f"Chat histories not found: {chat_path}. "
                 f"Run --phase generate first.")

    records = []
    with open(chat_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"\nLoaded {len(records)} chat histories from {chat_path}")

    out_path = os.path.join(
        ROOT_DIR, "data", "lvis_persona",
        "apol_to_pol_candidate_ranking.jsonl")

    race_keys = list(SENATE_RACES.keys())
    rng = random.Random(args.seed)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    async def _rank_one(idx: int) -> Dict[str, Any]:
        rec = dict(records[idx])
        chat_msgs = rec.get("messages", [])

        race_key = race_keys[idx % len(race_keys)]
        race = SENATE_RACES[race_key]

        system = CANDIDATE_SYSTEM.format(state=race["state"])
        user = CANDIDATE_USER.format(
            state=race["state"],
            dem_name=race["dem_name"],
            dem_desc=race["dem_desc"],
            rep_name=race["rep_name"],
            rep_desc=race["rep_desc"],
        )

        messages = list(chat_msgs)
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        async with sem:
            try:
                resp = await _call_chat(
                    client, args.model, messages, args.temperature, max_tokens=800)
            except Exception as exc:
                answer = f"[ERROR: {exc}]"
            else:
                answer = resp.choices[0].message.content.strip()

        rec["_response"] = answer
        rec["_domain"] = "candidate_ranking"
        rec["_race"] = {
            "key": race_key,
            "state": race["state"],
            "dem_name": race["dem_name"],
            "rep_name": race["rep_name"],
        }
        return rec

    tasks = [asyncio.create_task(_rank_one(i)) for i in range(len(records))]
    results = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Candidate ranking"):
        results.append(await fut)

    results.sort(key=lambda x: (GROUP_ORDER_ATP.index(x["group"]),
                                x["persona_id"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} rankings to: {out_path}")
    _print_candidate_summary(results)


def _print_candidate_summary(results: List[Dict[str, Any]]) -> None:
    from collections import Counter
    grouped = {g: [] for g in GROUP_ORDER_ATP}
    for r in results:
        grouped[r["group"]].append(r)

    print(f"\n  --- Candidate ranking summary ---")
    for g in GROUP_ORDER_ATP:
        count_dem = 0
        count_rep = 0
        count_err = 0
        for r in grouped[g]:
            resp = r.get("_response", "")
            if resp.startswith("[ERROR"):
                count_err += 1
                continue
            resp_lower = resp.lower()
            dem_name = r["_race"]["dem_name"].lower()
            rep_name = r["_race"]["rep_name"].lower()
            # Heuristic: which name appears first after "#1" or earliest mention
            dem_pos = resp_lower.find(dem_name)
            rep_pos = resp_lower.find(rep_name)
            if dem_pos >= 0 and (rep_pos < 0 or dem_pos < rep_pos):
                count_dem += 1
            else:
                count_rep += 1
        n = len(grouped[g])
        print(f"    {GROUP_DISPLAY_ATP[g]:<20s} "
              f"Dem #1: {count_dem}/{n}  Rep #1: {count_rep}/{n}  "
              f"Err: {count_err}/{n}")


# ═══════════════════════════════════════════════════════════════════
#  Phase 2: Senate speech generation + LLM judge
# ═══════════════════════════════════════════════════════════════════

async def phase_speech(args: argparse.Namespace) -> None:
    chat_path = args.chat_histories or os.path.join(
        ROOT_DIR, "data", "lvis_persona", "apol_to_pol_chat_histories.jsonl")

    if not os.path.exists(chat_path):
        sys.exit(f"Chat histories not found: {chat_path}. "
                 f"Run --phase generate first.")

    records = []
    with open(chat_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"\nLoaded {len(records)} chat histories from {chat_path}")

    out_path = os.path.join(
        ROOT_DIR, "data", "lvis_persona",
        "apol_to_pol_senate_speech.jsonl")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    async def _speech_one(idx: int) -> Dict[str, Any]:
        rec = dict(records[idx])
        chat_msgs = rec.get("messages", [])

        # Step 1: Generate speech
        gen_messages = list(chat_msgs)
        gen_messages.append({"role": "system", "content": SPEECH_SYSTEM})
        gen_messages.append({"role": "user", "content": SPEECH_USER})

        async with sem:
            try:
                resp = await _call_chat(
                    client, args.model, gen_messages, args.temperature,
                    max_tokens=800)
                speech_text = resp.choices[0].message.content.strip()
            except Exception as exc:
                speech_text = f"[ERROR: {exc}]"

        rec["_speech"] = speech_text
        rec["_domain"] = "senate_speech"

        # Step 2: Judge the speech
        if speech_text.startswith("[ERROR"):
            rec["_judge_eval"] = {"error": speech_text}
            return rec

        async with sem:
            try:
                judge_resp = await _call_chat(
                    client, args.model,
                    [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user",
                      "content": JUDGE_USER.format(speech=speech_text)}],
                    0.0, max_tokens=500,
                    response_format=JUDGE_JSON_SCHEMA,
                )
                eval_json = json.loads(
                    judge_resp.choices[0].message.content.strip())
            except Exception as exc:
                eval_json = {"error": str(exc)}

        rec["_judge_eval"] = eval_json
        return rec

    tasks = [asyncio.create_task(_speech_one(i))
             for i in range(len(records))]
    results = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Speech gen+judge"):
        results.append(await fut)

    results.sort(key=lambda x: (GROUP_ORDER_ATP.index(x["group"]),
                                x["persona_id"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} speeches to: {out_path}")
    _print_speech_summary(results)


def _print_speech_summary(results: List[Dict[str, Any]]) -> None:
    grouped = {g: [] for g in GROUP_ORDER_ATP}
    for r in results:
        grouped[r["group"]].append(r)

    dims = ["overall_lean", "economic_lean", "social_lean",
            "foreign_policy_lean", "institutional_trust"]

    print(f"\n  --- Speech lean summary (mean ± SD) ---")
    header = f"  {'Dimension':<22s}"
    for g in GROUP_ORDER_ATP:
        header += f" {GROUP_DISPLAY_ATP[g]:>22s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dim in dims:
        row = f"  {dim:<22s}"
        for g in GROUP_ORDER_ATP:
            vals = []
            for r in grouped[g]:
                ev = r.get("_judge_eval", {})
                if isinstance(ev, dict) and dim in ev:
                    vals.append(ev[dim])
            if vals:
                mu = sum(vals) / len(vals)
                sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
                row += f" {mu:>10.3f} ± {sd:.3f}"
            else:
                row += f" {'n/a':>22s}"
        print(row)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

async def main():
    args = parse_args()

    if args.phase in ("generate", "both"):
        await phase_generate_atp(args)

    if args.phase in ("candidates", "both"):
        await phase_candidates(args)

    if args.phase in ("speech", "both"):
        await phase_speech(args)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
