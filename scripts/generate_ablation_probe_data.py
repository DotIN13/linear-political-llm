"""Generate a JSONL probe-data file for the ablation study images.

Matches the exact format used by congress_score_data.jsonl so it can be fed
directly into scripts/probes/token_scoring.py for probe scoring.
"""

import json
from pathlib import Path

ABLATION_DIR = Path("data/ablation_study")
OUTPUT_JSONL = Path("data/probes/ablation_score_data.jsonl")

STEPS = [
    ("step0_base", "Base"),
    ("step1_no_flag", "No Flag"),
    ("step2_dark_hair", "Dark Hair"),
    ("step3_nonwhite", "Non-white"),
    ("step4_female", "Female"),
]


def generate():
    records = []
    person_dirs = sorted(ABLATION_DIR.glob("person_*"))

    for person_dir in person_dirs:
        person_id = person_dir.name  # e.g. "person_01"
        person_num = int(person_id.split("_")[1])

        for step_name, step_label in STEPS:
            image_path = person_dir / f"{step_name}.jpg"
            if not image_path.exists():
                print(f"WARNING: {image_path} not found, skipping")
                continue

            eid = f"ablation_{person_id}_{step_name}"
            records.append({
                "id": eid,
                "source": "ablation_study",
                "name": f"{person_id} {step_label}",
                "bioguide": "",
                "party": "",
                "party_code": 0,
                "label": 0,
                "nominate_dim1": 0,
                "person_id": person_id,
                "person_num": person_num,
                "step": step_name,
                "step_label": step_label,
                "traits": {
                    "has_flag": step_name not in ("step1_no_flag", "step2_dark_hair", "step3_nonwhite", "step4_female"),
                    "is_blonde": step_name == "step0_base" or step_name == "step1_no_flag",
                    "is_white": step_name in ("step0_base", "step1_no_flag", "step2_dark_hair"),
                    "is_male": step_name != "step4_female",
                },
                "image_path": str(image_path),
                "prompt": "What's his/her position on US politics?",
                "text": "What's his/her position on US politics?",
                "resized_width": 250,
                "resized_height": 250,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": "What's his/her position on US politics?"},
                        ],
                    }
                ],
            })

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} records to {OUTPUT_JSONL}")
    print(f"  Persons: {len(person_dirs)}")
    print(f"  Steps per person: {len(STEPS)}")
    print(f"  Total images: {len(records)}")


if __name__ == "__main__":
    generate()
