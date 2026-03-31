#!/usr/bin/env python3
"""
Generate one-sentence, policy-domain statements for sampled U.S. politicians
using OpenAI's chat completions API, in parallel.

Example:
    python generate_policy_statements.py \
        --csv ./data/HS116_members.csv \
        --samples-per-domain 20 \
        --model gpt-4o-mini \
        --out policy_statements_parallel.csv
"""

import argparse
import asyncio
import os
import random
from typing import Dict, List

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm


# ----------------------------
# Configuration / Defaults
# ----------------------------

# Policy queries (domain -> question). The prompt uses only the domain label.
POLICY_QUERIES = {
    "Abortion": "On a scale from 1 to 7, where 1 means strongly oppose and 7 means strongly favor, what is your opinion on making abortion always legal?",
    "Border Wall": "On a scale from 1 to 7, where 1 means strongly oppose and 7 means strongly favor, what is your opinion on building a wall on the U.S. border with Mexico?",
    "Transgender Bathroom": "On a scale from 1 to 7, where 1 means strongly oppose and 7 means strongly favor, what is your opinion on allowing transgender people to use public bathrooms matching their gender identity?",
    "Ukraine Military Aid": "On a scale from 1 to 7, where 1 means strongly oppose and 7 means strongly favor, what is your opinion on the U.S. giving weapons to help Ukraine fight Russia?",
    "Israel Military Aid": "On a scale from 1 to 7, where 1 means strongly oppose and 7 means strongly favor, what is your opinion on the U.S. giving military assistance to Israel?",
    "Gaza Humanitarian Aid": "On a scale from 1 to 7, where 1 means strongly oppose and 7 means strongly favor, what is your opinion on the U.S. providing humanitarian assistance to Palestinians in the Gaza Strip?",
}

MAX_CONCURRENT_REQUESTS = 10  # tune if you hit rate limits


# ----------------------------
# Data Loading & Prep
# ----------------------------

def load_and_prepare_members(csv_path: str) -> pd.DataFrame:
    """
    Load HS116 members CSV and construct a 'fullname' column matching the
    original notebook logic. Filters to rows with non-null `nominate_dim1`.
    Expects columns: ['bioname', 'nominate_dim1'].
    """
    df = pd.read_csv(csv_path)
    if "bioname" not in df.columns or "nominate_dim1" not in df.columns:
        raise ValueError("CSV must include 'bioname' and 'nominate_dim1' columns.")

    df = df.loc[pd.notnull(df.nominate_dim1)].reset_index(drop=True)

    # Name parsing (replicates the original logic)
    df["first_middle_name"] = df["bioname"].apply(lambda x: x.split(", ")[1])
    df["first_name"] = df["first_middle_name"].apply(
        lambda x: x.split(" ")[0] + " " if " " in x else x + " "
    )
    df["middle_name"] = df["first_middle_name"].apply(
        lambda x: x.split(" ")[1][0] + ". " if " " in x else ""
    )
    df["last_name"] = df["bioname"].apply(
        lambda x: x.split(", ")[0].lower().capitalize().strip()
    )
    df["fullname"] = df["first_name"] + df["middle_name"] + df["last_name"]
    return df


# ----------------------------
# OpenAI Helpers
# ----------------------------

def build_prompt(name: str, domain: str) -> str:
    """
    Construct a simple persona-style prompt for the specified name & domain.
    """
    return (
        f"You are {name}, a U.S. politician. "
        f"Write a realistic one-sentence statement that {name} might give about {domain}."
    )


async def query_policy_statement(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    name: str,
    domain: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, str]:
    """
    Make one OpenAI chat completion request for a (name, domain) pair.
    Returns a dict with (name, domain, response).
    """
    prompt = build_prompt(name, domain)
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content.strip()
            return {"name": name, "domain": domain, "response": content}
        except Exception as e:
            return {"name": name, "domain": domain, "response": f"ERROR: {str(e)}"}



async def generate_statements(
    csv_path: str,
    samples_per_domain: int,
    output_csv: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
) -> None:
    """
    Orchestrates loading data, sampling names per domain, and running
    concurrent OpenAI requests. Writes results to CSV and prints to stdout.
    """
    # Load API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")
    client = AsyncOpenAI(api_key=api_key)

    # Load & prep data
    df = load_and_prepare_members(csv_path)

    # Optional seed for reproducibility across runs
    if seed is not None:
        random.seed(seed)

    # Build tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks: List[asyncio.Task] = []

    for domain in POLICY_QUERIES.keys():
        random_state = random.randint(0, 42)
        sampled_names = df["fullname"].sample(samples_per_domain, random_state=random_state).tolist()
        for name in sampled_names:
            tasks.append(
                asyncio.create_task(
                    query_policy_statement(
                        client=client,
                        semaphore=semaphore,
                        name=name,
                        domain=domain,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
            )

    # Progress bar with as_completed
    results: List[Dict[str, str]] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating statements"):
        r = await coro
        results.append(r)

    # Print to stdout
    for r in results:
        print(f"{r['name']} on {r['domain']}:\n{r['response']}\n")

    # Write to CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved {len(results)} rows to {output_csv}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate policy statements for sampled U.S. politicians.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to HS116_members.csv (must include 'bioname' and 'nominate_dim1' columns).",
    )
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=20,
        help="Number of names to sample per policy domain.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for responses.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Max tokens for each generated response.",
    )
    parser.add_argument(
        "--out",
        default="policy_statements_parallel.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        generate_statements(
            csv_path=args.csv,
            samples_per_domain=args.samples_per_domain,
            output_csv=args.out,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
