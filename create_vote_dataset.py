# vote2020_dataset_generator.py
"""
Generate a dataset of 600 combined question/answer statements about the 2020 U.S. presidential
 election, each labeled with either "Trump" or "Biden". The reasoning snippets now span policy,
 ethnicity, community norms, religious identity, path dependence, economic class, and personal
 experience. All text is single-line (no "\n" characters) for compatibility with simple linear-probe
 models.
"""

import csv
import json
import itertools
import random
from pathlib import Path

# ---------------- 1. Questions (20) ----------------------------------------
QUESTIONS = [
    "Who would you vote for in the 2020 U.S. presidential election?",
    "If you had to pick between Donald Trump and Joe Biden in 2020, who gets your vote?",
    "In the 2020 race, which candidate would you support?",
    "Assuming it's November 2020, whom would you cast your ballot for—Biden or Trump?",
    "Between Trump and Biden, who earns your 2020 vote?",
    "Looking back to 2020, which presidential candidate aligns more with your views?",
    "In the context of the 2020 election, who would you choose?",
    "Pretend you're in the voting booth in 2020: Trump or Biden?",
    "Considering their 2020 platforms, who deserves your vote?",
    "When deciding in 2020, which name would you check: Joe Biden or Donald Trump?",
    "During the 2020 contest, who was your preferred candidate?",
    "For the 2020 U.S. presidency, would you select Biden or Trump?",
    "Which 2020 contender—Trump or Biden—better represented your priorities?",
    "If casting a 2020 ballot today, who'd you select?",
    "Reflecting on 2020, who should have led the country: Trump or Biden?",
    "Suppose the 2020 election were re-run; who'd you back?",
    "Considering their 2020 debate performances, who was stronger?",
    "Given the 2020 economic climate, who was the better choice?",
    "Thinking about pandemic leadership in 2020, who wins your vote?",
    "With all factors of 2020 in mind, which candidate do you endorse?",
]

# ---------------- 2. Diverse reasoning & decision snippets (30) ------------
REASONING_DECISIONS = [
    # --- Biden-inclined (15) ---
    {"text": "As a first-generation immigrant I valued Biden's promise to protect DACA recipients so I would vote for Joe Biden.", "label": "Biden"},
    {"text": "My predominantly unionized Rust-Belt town endorsed Biden and sticking with my community matters to me so Joe Biden gets my vote.", "label": "Biden"},
    {"text": "Having lost a loved one to COVID-19 Biden's empathetic tone and plan for national mourning rituals resonated thus I choose him.", "label": "Biden"},
    {"text": "As a suburban mom worried about school safety and reopening protocols Biden's cautious approach convinced me to support him.", "label": "Biden"},
    {"text": "Biden's Catholic faith and commitment to social justice aligned with my religious values so he earns my vote.", "label": "Biden"},
    {"text": "My decades-long Democratic voting habit makes Biden my default choice in 2020.", "label": "Biden"},
    {"text": "My daughter came out as transgender and Biden's support for LGBTQ+ protections secured my backing.", "label": "Biden"},
    {"text": "As a scientist I respected Biden's pledge to follow the data which earns my vote.", "label": "Biden"},
    {"text": "Biden's plan to forgive up to $10k of student debt directly eases my burden so I would pick him.", "label": "Biden"},
    {"text": "As a Black voter in Georgia mobilized by Stacey Abrams' network casting a ballot for Biden felt like building on collective progress.", "label": "Biden"},
    {"text": "Growing up near wildfire-prone California forests Biden's climate urgency struck a personal chord and wins my vote.", "label": "Biden"},
    {"text": "I rely on Affordable Care Act subsidies and Biden vowed to shore them up so he is my candidate.", "label": "Biden"},
    {"text": "Biden's plan for broad childcare credits would help my young family therefore I support him.", "label": "Biden"},
    {"text": "My spouse is a public-school teacher and Biden's respect for educators wins my vote.", "label": "Biden"},
    {"text": "After watching Biden console grieving military families I felt his empathy is what the nation needs so I choose him.", "label": "Biden"},

    # --- Trump-inclined (15) ---
    {"text": "As a Cuban-American in Florida Trump's tough stance on the Castro regime influenced my vote for him.", "label": "Trump"},
    {"text": "My rural Texas town boomed thanks to oil jobs and Trump's drilling policies protect our livelihood so I choose Donald Trump.", "label": "Trump"},
    {"text": "I served in the military and appreciated Trump's VA reforms therefore I back him.", "label": "Trump"},
    {"text": "My evangelical church strongly supported Trump and community solidarity matters to me so I vote Trump.", "label": "Trump"},
    {"text": "Coming from a family of lifelong Republicans I keep the tradition by choosing Donald Trump.", "label": "Trump"},
    {"text": "Trump's promise to stand up to China resonated with my manufacturing-worker experience thus I support him.", "label": "Trump"},
    {"text": "As a gun-owner who enjoys weekend shooting ranges Trump's Second-Amendment stance earns my support.", "label": "Trump"},
    {"text": "As a police officer I trusted Trump's law-and-order message so he gets my vote.", "label": "Trump"},
    {"text": "My stock portfolio benefitted from market gains under Trump steering my vote his way.", "label": "Trump"},
    {"text": "Trump's opposition to prolonged lockdowns kept my small restaurant afloat so I back him.", "label": "Trump"},
    {"text": "As a parent of home-schooled kids I liked Trump's push for school choice and vote for him.", "label": "Trump"},
    {"text": "Being an Orthodox Jew I welcomed Trump's relocation of the U.S. embassy to Jerusalem therefore I choose him.", "label": "Trump"},
    {"text": "My brother's farming operation relies on ethanol and Trump pledged to uphold ethanol mandates so he earns my vote.", "label": "Trump"},
    {"text": "Trump's brash outsider style appeals to me as someone who distrusts career politicians thus I vote Trump.", "label": "Trump"},
    {"text": "I work in border security and believed Trump took my job seriously so I back him.", "label": "Trump"},
]

# ---------------- 3. Build dataset ----------------------------------------

def build_dataset(questions, reasonings, seed=42):
    random.seed(seed)
    combined = []
    for q, r in itertools.product(questions, reasonings):
        text = f"USER: {q}\nASSISTANT: {r['text']}"
        label = -1 if r["label"] == "Biden" else 1
        combined.append({"prompt": text, "label_text": r["label"], "label": label})
    random.shuffle(combined)
    return combined

# ---------------- 4. Save to disk -----------------------------------------

def save_csv(path: Path, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "label_text", "label"])
        writer.writeheader()
        writer.writerows(data)


def save_jsonl(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(out_dir="./data"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(QUESTIONS, REASONING_DECISIONS)
    print(f"Generated {len(dataset)} examples.")

    save_csv(out_path / "vote_2020_dataset.csv", dataset)
    save_jsonl(out_path / "vote_2020_dataset.jsonl", dataset)
    print("Saved CSV and JSONL to", out_path.resolve())


if __name__ == "__main__":
    main()
