"""Interactive CLI for LLM-based tweet profiling of political figures.

Usage:
    python profile_tweets.py [--parquet PARQUET] [--n_tweets N] [--model MODEL]
    python profile_tweets.py --user tedcruz

Type a screen_name to profile; type 'quit' to exit.
"""

import argparse
import random
import sys
from collections import defaultdict

import pandas as pd
from google import genai


def load_tweets(parquet_path):
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df):,} tweets, {df['screen_name'].nunique():,} unique users")

    tweets_by_user = defaultdict(list)
    for row in df.itertuples():
        sn = row.screen_name
        if isinstance(sn, str) and sn.strip():
            tweets_by_user[sn].append(row.text)
    return tweets_by_user


def build_prompt(tweets, screen_name):
    tweets_text = "\n".join(f"- \"{t}\"" for t in tweets)
    return f"""You are a political analyst. Below are tweets from a U.S. political figure with handle @{screen_name}.

Tweets:
{tweets_text}

Please provide a detailed profile of this person based solely on their tweets. Include:

1. **Political affiliation** — party, ideology (liberal/conservative/moderate), and confidence.
2. **Key issues** — what topics and policies they care about most.
3. **Communication style** — tone, rhetorical strategies, engagement style.
4. **Personality traits** — what their tweets reveal about them as a person.
5. **Target audience** — who they seem to be speaking to.
6. **Summary** — a 1-2 sentence overall profile.

Be specific and evidence-based. If the tweets are ambiguous, note that."""


def resolve_user(query, tweets_by_user):
    q = query.lower()
    exact = [u for u in tweets_by_user if u.lower() == q]
    if exact:
        return exact[0]
    matches = [u for u in tweets_by_user if q in u.lower()]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  Multiple matches: {', '.join(matches[:10])}")
        print(f"  Using first match: {matches[0]}")
    return matches[0]


def profile_user(screen_name, tweets_by_user, client, args, interactive=True):
    screen_name = resolve_user(screen_name, tweets_by_user)
    if screen_name is None:
        print(f"  No user found matching query.")
        return

    all_tweets = tweets_by_user[screen_name]
    n_sample = min(args.n_tweets, len(all_tweets))
    sampled = random.sample(all_tweets, n_sample)

    prompt = build_prompt(sampled, screen_name)
    print(f"\n  Profiling @{screen_name} using {n_sample}/{len(all_tweets)} tweets ...")

    response = client.models.generate_content(
        model=args.model,
        contents=prompt,
    )

    print(f"\n{'='*70}")
    print(response.text)
    print(f"{'='*70}")
    return screen_name


def main():
    parser = argparse.ArgumentParser(description="LLM-based tweet profiling")
    parser.add_argument("--parquet", default="../trump-vector/artifacts/congress_tweets.parquet")
    parser.add_argument("--n_tweets", type=int, default=20,
                        help="Number of tweets to sample per user")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model to use")
    parser.add_argument("--user", default=None,
                        help="Profile a specific user (non-interactive mode)")
    args = parser.parse_args()

    tweets_by_user = load_tweets(args.parquet)
    client = genai.Client(api_key=None)

    if args.user:
        profile_user(args.user, tweets_by_user, client, args)
        return

    print("\nEnter a screen_name to profile (type 'quit' to exit):")

    try:
        while True:
            screen_name = input("\n@").strip()
            if screen_name.lower() == "quit":
                print("Goodbye.")
                break

            if not screen_name:
                continue

            profile_user(screen_name, tweets_by_user, client, args)

    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
