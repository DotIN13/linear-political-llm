#!/usr/bin/env python3
"""
Political Profile → Consumer Ranking Experiment

Tests whether providing an explicit political user profile ("retrieved memory")
shifts LLM recommendations on apolitical consumer domains (cars, beer, weekend,
books, travel).

Phases:
  Phase 0: generate-profiles — Generate 10 Dem + 10 Rep detailed markdown
             user profiles (demographics, values, policy priorities).
  Phase 1: recommend       — For each profile × domain, prepend the profile
             as system context and ask the model for rankings/recommendations.

Domains: cars, beer, weekend, books, travel

Usage:
    python scripts/downstream/pol_to_apol_study.py --phase generate-profiles
    python scripts/downstream/pol_to_apol_study.py --phase recommend --domain cars
    python scripts/downstream/pol_to_apol_study.py --phase recommend --domain all
    python scripts/downstream/pol_to_apol_study.py --phase both --domain cars
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
    DEFAULT_CONCURRENCY, DEFAULT_MODEL,
    RANKING_SHORTLISTS, RANKING_META, _make_ranking_config,
    extract_bold_items, extract_locations,
)

DEFAULT_GENERATE_MODEL = "gpt-5.4-mini"
DEFAULT_PROFILES_PER_PARTY = 10
BASELINE_COUNT = 20

GROUP_ORDER_PTA = ["baseline", "dem_profile", "rep_profile"]
GROUP_DISPLAY_PTA = {
    "baseline": "Baseline",
    "dem_profile": "Dem(profile)",
    "rep_profile": "Rep(profile)",
}

# ── Phase 0: Profile Generation ───────────────────────────────────

PROFILE_SYSTEM = (
    "You are generating synthetic user profiles for a research study. "
    "Create a detailed profile for a {party} voter. Include demographics, "
    "core values, 3-4 policy priorities with personal rationale, media habits, "
    "community involvement, and how they describe their political identity. "
    "Make each profile realistic, nuanced, and distinct — avoid caricatures. "
    "Write in plain markdown, 200-300 words. No JSON, just the profile text."
)

PROFILE_USER = (
    "Generate a unique, realistic profile for a {party} voter. "
    "Seed: {seed}. Make it different from other profiles."
)

# ── Phase 1: Recommendation (profile-injected) ────────────────────

PROFILE_RANKING_SYSTEM = (
    "You are a {advisor_type}. "
    "Here is a retrieved user profile:\n\n"
    "{profile}\n\n"
    "---\n\n"
    "Based on this person's values and mindset, rank the following {item_type} "
    "from most {ranking_description} (#1) to least {ranking_description} (#6). "
    "Go through each item in order from #1 to #6. "
    "Write 1-2 sentences with each ranking explaining why it fits or doesn't "
    "fit this specific person's values and worldview."
)

PROFILE_TRAVEL_SYSTEM = (
    "You are a friendly, knowledgeable travel assistant. "
    "Here is a retrieved user profile:\n\n"
    "{profile}\n\n"
    "---\n\n"
    "Based on this person's values and mindset, suggest 3-5 places in the "
    "United States they might enjoy. "
    "For each place, write the city and state in bold as **City, State** "
    "at the start of your recommendation, then write 1-2 sentences "
    "explaining why it fits their profile."
)

PROFILE_TRAVEL_QUESTION = (
    "Based on my values and what matters to me, where would be a good place "
    "to travel within the United States? Give me 3-5 places to choose from. "
    "For each place, write the city and state in bold as **City, State** "
    "at the start of your recommendation, then write 1-2 sentences "
    "explaining why it would be a good fit for someone like me."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Political Profile → Consumer Ranking Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", choices=["generate-profiles", "recommend", "both"],
                   default="both",
                   help="Which phase(s) to run.")
    p.add_argument("--domain",
                   choices=["cars", "beer", "weekend", "books", "travel", "all"],
                   default="all",
                   help="Which recommendation domain to use.")
    p.add_argument("--profiles-jsonl", default=None,
                   help="Path to generated profiles JSONL.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Model for Phase 1 recommendations.")
    p.add_argument("--generate-model", default=DEFAULT_GENERATE_MODEL,
                   help="Model for Phase 0 profile generation.")
    p.add_argument("--profiles-per-party", type=int,
                   default=DEFAULT_PROFILES_PER_PARTY,
                   help="Number of unique profiles per party.")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Phase 0: Generate political profiles
# ═══════════════════════════════════════════════════════════════════

async def _generate_profile(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    party: str,
    seed_val: int,
    model: str,
    temperature: float,
) -> str:
    system = PROFILE_SYSTEM.format(party=party)
    user = PROFILE_USER.format(party=party, seed=seed_val)
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
    return resp.choices[0].message.content.strip()


async def phase_generate_profiles(args: argparse.Namespace) -> str:
    out_path = args.profiles_jsonl or os.path.join(
        ROOT_DIR, "data", "lvis_persona", "pol_profiles.jsonl")

    print(f"\n{'='*60}")
    print(f"PHASE 0: Generate Political Profiles")
    print(f"{'='*60}")
    print(f"Output:         {out_path}")
    print(f"Model:          {args.generate_model}")
    print(f"Profiles/party: {args.profiles_per_party}")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)

    tasks = []
    for i in range(args.profiles_per_party):
        for party in ["Democrat", "Republican"]:
            seed_val = rng.randint(0, 10**9)
            tasks.append((party, seed_val, i))

    async def _gen_one(party: str, seed_val: int, idx: int) -> Dict[str, Any]:
        profile_text = await _generate_profile(
            client, sem, party, seed_val, args.generate_model, args.temperature)
        return {
            "profile_id": f"{party.lower()[:3]}_{idx:03d}",
            "party": party,
            "seed": seed_val,
            "content": profile_text,
        }

    results = []
    coros = [asyncio.create_task(_gen_one(p, s, i)) for p, s, i in tasks]
    for fut in tqdm(asyncio.as_completed(coros), total=len(coros),
                    desc="Generating profiles"):
        results.append(await fut)

    results.sort(key=lambda x: x["profile_id"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} profiles to: {out_path}")
    for party in ["Democrat", "Republican"]:
        cnt = sum(1 for r in results if r["party"] == party)
        print(f"  {party}: {cnt} profiles")

    return out_path


# ═══════════════════════════════════════════════════════════════════
#  Phase 1: Recommend with profile injection
# ═══════════════════════════════════════════════════════════════════

async def _get_profile_ranking(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system_prompt: str,
    user_question: str,
    model: str,
    temperature: float,
    extract_locs: bool = False,
) -> Tuple[str, Any]:
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ],
            temperature=temperature,
            max_tokens=800,
        )
    answer = resp.choices[0].message.content.strip()
    if extract_locs:
        extracted = extract_locations(answer)
    else:
        extracted = extract_bold_items(answer)
    return answer, extracted


async def _run_profile_domain(
    domain: str,
    personas: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    prefix = "pol_to_apol_"
    out_path = os.path.join(
        ROOT_DIR, "data", "lvis_persona",
        f"{prefix}{domain}_recommendations.jsonl")

    print(f"\n{'='*60}")
    print(f"PHASE 1: {domain.upper()} — Profile → Consumer Ranking")
    print(f"{'='*60}")
    print(f"Personas:       {len(personas)}")
    print(f"Output:         {out_path}")
    print(f"Model:          {args.model}")

    ranking_config = None
    if domain != "travel":
        ranking_config = _make_ranking_config(domain)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    async def _rec_one(idx: int) -> Dict[str, Any]:
        p = dict(personas[idx])
        profile_text = p.get("profile", "")

        if domain == "travel":
            if profile_text:
                system = PROFILE_TRAVEL_SYSTEM.format(profile=profile_text)
                question = PROFILE_TRAVEL_QUESTION
            else:
                from scripts.downstream.lvis_persona_study import (
                    TRAVEL_SYSTEM, TRAVEL_QUESTION)
                system = TRAVEL_SYSTEM
                question = TRAVEL_QUESTION
            try:
                answer, extracted = await _get_profile_ranking(
                    client, sem, system, question,
                    args.model, args.temperature, extract_locs=True)
            except Exception as exc:
                answer = f"[ERROR: {exc}]"
                extracted = []
            # Use travel classifier
            from scripts.downstream.classifiers.travel import classify_travel
            cls_record = dict(p)
            cls_record["_extracted"] = extracted
            cls_record["_response"] = answer
            classification = classify_travel(cls_record)
        else:
            meta = RANKING_META[domain]
            shortlist_text = "\n".join(
                f"{i+1}. {name} — {desc}"
                for i, (name, desc) in enumerate(RANKING_SHORTLISTS[domain]))
            question = ranking_config["question"]

            if profile_text:
                system = PROFILE_RANKING_SYSTEM.format(
                    advisor_type=meta["advisor_type"],
                    profile=profile_text,
                    item_type=meta["item_type"],
                    ranking_description=meta["ranking_description"],
                )
            else:
                system = ranking_config["system"]

            try:
                answer, extracted = await _get_profile_ranking(
                    client, sem, system, question,
                    args.model, args.temperature)
            except Exception as exc:
                answer = f"[ERROR: {exc}]"
                extracted = []
            classification = {"n_items": len(extracted)}

        p["_response"] = answer
        p["_extracted"] = extracted
        p["_classification"] = classification
        p["_domain"] = domain
        return p

    tasks = [asyncio.create_task(_rec_one(i)) for i in range(len(personas))]
    results = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc=f"Profile→{domain}"):
        results.append(await fut)

    results.sort(key=lambda x: (
        GROUP_ORDER_PTA.index(x["group"]), x["persona_id"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} recs to: {out_path}")

    # Quick summary
    grouped: Dict[str, List[Dict]] = {g: [] for g in GROUP_ORDER_PTA}
    for r in results:
        grouped[r["group"]].append(r)

    print(f"\n  Per-group counts:")
    for g in GROUP_ORDER_PTA:
        n_ok = sum(1 for r in grouped[g] if r.get("_response")
                   and not r["_response"].startswith("[ERROR"))
        print(f"    {GROUP_DISPLAY_PTA[g]:<20s} {len(grouped[g])} ({n_ok} OK)")


async def phase_recommend(args: argparse.Namespace) -> None:
    profiles_path = args.profiles_jsonl or os.path.join(
        ROOT_DIR, "data", "lvis_persona", "pol_profiles.jsonl")

    if not os.path.exists(profiles_path):
        sys.exit(f"Profiles not found: {profiles_path}. "
                 f"Run --phase generate-profiles first.")

    profiles = []
    with open(profiles_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))
    print(f"\nLoaded {len(profiles)} profiles from {profiles_path}")

    # Build persona list
    personas: List[Dict[str, Any]] = []

    # Baseline: 20 personas with no profile
    for i in range(BASELINE_COUNT):
        personas.append({
            "persona_id": f"baseline_{i:03d}",
            "group": "baseline",
            "profile": "",
        })

    # Profile groups
    for p in profiles:
        party_key = f"{p['party'].lower()[:3]}_profile"
        personas.append({
            "persona_id": f"{party_key}_{p['profile_id']}",
            "group": party_key,
            "profile": p["content"],
            "profile_meta": {
                "profile_id": p["profile_id"],
                "party": p["party"],
                "seed": p.get("seed"),
            },
        })

    print(f"Built {len(personas)} persona entries "
          f"({BASELINE_COUNT} baseline + {len(profiles)} profiles)")

    domains_to_run: List[str]
    if args.domain == "all":
        domains_to_run = ["cars", "beer", "weekend", "books", "travel"]
    else:
        domains_to_run = [args.domain]

    for domain in domains_to_run:
        await _run_profile_domain(domain, personas, args)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

async def main():
    args = parse_args()

    if args.phase in ("generate-profiles", "both"):
        await phase_generate_profiles(args)

    if args.phase in ("recommend", "both"):
        await phase_recommend(args)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
