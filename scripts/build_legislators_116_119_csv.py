import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_PATH = ROOT / "data" / "legislators-current.json"
HISTORICAL_PATH = ROOT / "data" / "legislators-historical.json"
OUT_PATH = ROOT / "data" / "legislators_116_119.csv"

CONGRESSES = {
    116: (date(2019, 1, 3), date(2021, 1, 3)),
    117: (date(2021, 1, 3), date(2023, 1, 3)),
    118: (date(2023, 1, 3), date(2025, 1, 3)),
    119: (date(2025, 1, 3), date(2027, 1, 3)),
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def chamber(term_type: str) -> str:
    if term_type == "sen":
        return "senate"
    if term_type == "rep":
        return "house"
    return term_type


def overlap(start_a: date, end_a: date, start_b: date, end_b: date):
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    if start >= end:
        return None
    return start, end


def iter_legislators():
    for src_label, path in (("current", CURRENT_PATH), ("historical", HISTORICAL_PATH)):
        with path.open() as f:
            for person in json.load(f):
                yield src_label, person


def person_name(name_obj: dict) -> str:
    if name_obj.get("official_full"):
        return name_obj["official_full"]
    first = name_obj.get("first", "")
    middle = name_obj.get("middle", "")
    last = name_obj.get("last", "")
    suffix = name_obj.get("suffix", "")
    return " ".join(part for part in (first, middle, last, suffix) if part)


def build_rows():
    rows = []
    for src_label, person in iter_legislators():
        person_id = person.get("id", {})
        name_obj = person.get("name", {})
        bio = person.get("bio", {})
        bioguide = person_id.get("bioguide")

        if not bioguide:
            continue

        for term in person.get("terms", []):
            t_start = parse_date(term["start"])
            t_end = parse_date(term["end"])

            for congress_num, (c_start, c_end) in CONGRESSES.items():
                inter = overlap(t_start, t_end, c_start, c_end)
                if inter is None:
                    continue

                overlap_start, overlap_end = inter
                rows.append(
                    {
                        "bioguide": bioguide,
                        "name": person_name(name_obj),
                        "first": name_obj.get("first", ""),
                        "last": name_obj.get("last", ""),
                        "gender": bio.get("gender", ""),
                        "birthday": bio.get("birthday", ""),
                        "source": src_label,
                        "congress": congress_num,
                        "chamber": chamber(term.get("type", "")),
                        "state": term.get("state", ""),
                        "district": term.get("district", ""),
                        "senate_class": term.get("class", ""),
                        "party": term.get("party", ""),
                        "term_start": term.get("start", ""),
                        "term_end": term.get("end", ""),
                        "congress_start": c_start.isoformat(),
                        "congress_end": c_end.isoformat(),
                        "overlap_start": overlap_start.isoformat(),
                        "overlap_end": overlap_end.isoformat(),
                    }
                )

    # Remove duplicates that can occur when the same person appears in both files.
    dedup = {}
    for row in rows:
        key = (
            row["bioguide"],
            row["congress"],
            row["chamber"],
            row["state"],
            row["district"],
            row["term_start"],
            row["term_end"],
        )
        dedup[key] = row

    return sorted(
        dedup.values(),
        key=lambda r: (r["congress"], r["chamber"], r["state"], r["last"], r["first"], r["bioguide"]),
    )


def main():
    rows = build_rows()
    fields = [
        "bioguide",
        "name",
        "first",
        "last",
        "gender",
        "birthday",
        "source",
        "congress",
        "chamber",
        "state",
        "district",
        "senate_class",
        "party",
        "term_start",
        "term_end",
        "congress_start",
        "congress_end",
        "overlap_start",
        "overlap_end",
    ]

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
