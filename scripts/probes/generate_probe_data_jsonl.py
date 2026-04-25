#!/usr/bin/env python3
"""
Generate JSONL manifests for textual and combined ideology probes.

Outputs records with fields expected by probes CLI:
- textual: text, label
- combined: text, label, image_path

Default sources:
- data/HS116_members.csv
- data/legislators_116_119_with_twitter.csv
- data/HS116_members.csv
- data/congress_images/<BIOGUIDE>.jpg
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ProbeRecord:
    text: str
    label: float
    name: str
    bioguide: str
    source: str
    messages: List[Dict[str, Any]]
    image_path: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build textual/combined JSONL probe datasets using vl_vision_probe prompt schemes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hs-csv", default="data/HS116_members.csv")
    p.add_argument("--legislators-csv", default="data/legislators_116_119_with_twitter.csv")
    p.add_argument("--image-dir", default="data/congress_images")
    p.add_argument("--output-dir", default="data/probes")
    p.add_argument("--textual-output", default="textual_ideology.jsonl")
    p.add_argument("--combined-output", default="combined_ideology.jsonl")
    p.add_argument("--source-tag", default="hs116_congress_images")
    return p.parse_args()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_probe_text(name: str) -> str:
    # Mirrors vl_vision_probe.ipynb textual/combined construction.
    clean = str(name).strip()
    if not clean:
        clean = "this politician"
    return f"What would {clean} say in a policy interview?"


def build_messages(text: str, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    content: List[Dict[str, str]] = []
    if image_path is not None:
        # Keep image before text to match combined scheme in vl_vision_probe.ipynb.
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def build_bioguide_to_name(legislators_rows: Iterable[Dict[str, str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in legislators_rows:
        name = row.get("name", "").strip()
        bg = row.get("bioguide", "").strip().upper()
        if not name or not bg:
            continue
        lookup[bg] = name
    return lookup


def build_bioguide_to_nominate(hs_rows: Iterable[Dict[str, str]]) -> Dict[str, float]:
    lookup: Dict[str, float] = {}
    for row in hs_rows:
        bg = row.get("bioguide_id", "").strip().upper()
        raw = row.get("nominate_dim1", "").strip()
        if not bg or not raw:
            continue
        try:
            lookup[bg] = float(raw)
        except ValueError:
            continue
    return lookup


def find_image_path(image_dir: str, bioguide: str) -> Optional[str]:
    path = os.path.join(image_dir, f"{bioguide}.jpg")
    if os.path.exists(path):
        return path
    return None


def write_jsonl(path: str, records: Iterable[ProbeRecord], include_image: bool) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            payload = {
                "text": r.text,
                "label": r.label,
                "name": r.name,
                "bioguide": r.bioguide,
                "source": r.source,
                "messages": r.messages,
            }
            if include_image:
                payload["image_path"] = r.image_path
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()

    hs_rows = read_csv_rows(args.hs_csv)
    legislators_rows = read_csv_rows(args.legislators_csv)

    bg_to_nominate = build_bioguide_to_nominate(hs_rows)
    bg_to_name = build_bioguide_to_name(legislators_rows)

    textual_records: List[ProbeRecord] = []
    combined_records: List[ProbeRecord] = []

    missing_bioguide = 0
    missing_nominate = 0
    missing_image = 0

    for row in hs_rows:
        bioguide = str(row.get("bioguide_id", "")).strip().upper()
        if not bioguide:
            missing_bioguide += 1
            continue

        label = bg_to_nominate.get(bioguide)
        if label is None:
            missing_nominate += 1
            continue

        image_path = find_image_path(args.image_dir, bioguide)
        if image_path is None:
            missing_image += 1
            continue

        name = bg_to_name.get(bioguide, str(row.get("bioname", "")).strip())
        text_value = make_probe_text(name)

        textual_records.append(
            ProbeRecord(
                text=text_value,
                label=label,
                name=name,
                bioguide=bioguide,
                source=args.source_tag,
                messages=build_messages(text=text_value),
                image_path=image_path,
            )
        )

        combined_records.append(
            ProbeRecord(
                text=text_value,
                label=label,
                name=name,
                bioguide=bioguide,
                source=args.source_tag,
                messages=build_messages(text=text_value, image_path=image_path),
                image_path=image_path,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    textual_path = os.path.join(args.output_dir, args.textual_output)
    combined_path = os.path.join(args.output_dir, args.combined_output)

    textual_n = write_jsonl(textual_path, textual_records, include_image=False)
    combined_n = write_jsonl(combined_path, combined_records, include_image=True)

    print(f"Wrote textual manifest : {textual_path} ({textual_n} rows)")
    print(f"Wrote combined manifest: {combined_path} ({combined_n} rows)")
    print("--- Match summary ---")
    print(f"HS rows read           : {len(hs_rows)}")
    print(f"Missing bioguide       : {missing_bioguide}")
    print(f"Missing NOMINATE score : {missing_nominate}")
    print(f"Missing image          : {missing_image}")


if __name__ == "__main__":
    main()
