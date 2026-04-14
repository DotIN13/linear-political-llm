import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = ROOT / "data" / "legislators_116_119.csv"
DEFAULT_SOCIAL_PATH = ROOT / "data" / "legislators-social-media.json"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "legislators_116_119_with_twitter.csv"


def load_twitter_handles(path: Path) -> dict[str, str]:
    with path.open() as handle:
        records = json.load(handle)

    twitter_by_bioguide = {}
    for record in records:
        bioguide = record.get("id", {}).get("bioguide")
        twitter = record.get("social", {}).get("twitter")
        if bioguide and twitter:
            twitter_by_bioguide[bioguide] = twitter

    return twitter_by_bioguide


def output_path_for_args(output: str | None, in_place: bool, input_path: Path) -> Path:
    if in_place:
        return input_path
    if output:
        return Path(output)
    return DEFAULT_OUTPUT_PATH


def add_twitter_column(input_path: Path, social_path: Path, output_path: Path) -> tuple[int, int]:
    twitter_by_bioguide = load_twitter_handles(social_path)

    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "twitter" not in fieldnames:
        insert_at = fieldnames.index("bioguide") + 1 if "bioguide" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "twitter")

    matched_rows = 0
    for row in rows:
        twitter = twitter_by_bioguide.get(row.get("bioguide", ""), "")
        row["twitter"] = twitter
        if twitter:
            matched_rows += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), matched_rows


def main():
    parser = argparse.ArgumentParser(
        description="Add a twitter column to legislators_116_119.csv using legislators-social-media.json."
    )
    parser.add_argument("--input", default=str(DEFAULT_CSV_PATH), help="Path to the source CSV file.")
    parser.add_argument("--social", default=str(DEFAULT_SOCIAL_PATH), help="Path to the social-media JSON file.")
    parser.add_argument("--output", help="Path to the output CSV file.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV instead of writing a separate output file.",
    )
    args = parser.parse_args()

    if args.in_place and args.output:
        parser.error("--output and --in-place cannot be used together")

    input_path = Path(args.input)
    social_path = Path(args.social)
    output_path = output_path_for_args(args.output, args.in_place, input_path)

    total_rows, matched_rows = add_twitter_column(input_path, social_path, output_path)
    print(f"Wrote {total_rows} rows to {output_path}")
    print(f"Matched twitter handles for {matched_rows} rows")
    print(f"Rows without twitter handles: {total_rows - matched_rows}")


if __name__ == "__main__":
    main()