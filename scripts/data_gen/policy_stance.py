#!/usr/bin/env python3

"""
Combine policy stances with specific U.S. policy items into prompt strings.

Output columns:
- domain
- policy
- stance
- target_score
- prompt
"""

from typing import Dict, List
from datetime import datetime, timezone
import pandas as pd


# ----------------------------
# Policies
# ----------------------------

POLICIES: List[Dict[str, str]] = [
    {"domain": "Abortion", "policy": "State authority to restrict or ban abortion"},
    {"domain": "Abortion", "policy": "Opposing ballot initiatives that expand abortion access"},
    {"domain": "Abortion", "policy": "Restricting access to medication abortion and telehealth prescribing"},
    {"domain": "Abortion", "policy": "Recognizing embryos as having legal protections in IVF and fertility care"},
    {"domain": "Abortion", "policy": "Limiting federal mandates requiring abortion care in emergency settings"},

    {"domain": "Gender", "policy": "Banning gender-affirming medical care for minors"},
    {"domain": "Gender", "policy": "Restricting or regulating gender-affirming medical care for adults"},
    {"domain": "Gender", "policy": "Requiring transgender students to compete in sports based on biological sex"},
    {"domain": "Gender", "policy": "Restricting bathroom access based on biological sex"},
    {"domain": "Gender", "policy": "Limiting changes to legal documents based on gender identity"},
    {"domain": "Gender", "policy": "Expanding parental notification and consent requirements in schools"},
    {"domain": "Gender", "policy": "Limiting LGBTQ+ nondiscrimination laws with religious liberty carve-outs"},
    {"domain": "Gender", "policy": "Broad religious exemptions allowing refusal of services related to LGBTQ+ issues"},

    {"domain": "Immigration", "policy": "Expanding physical barriers and fencing at the U.S.–Mexico border"},
    {"domain": "Immigration", "policy": "Tightening asylum eligibility and accelerating removals"},
    {"domain": "Immigration", "policy": "Expanded deportations for undocumented immigrants regardless of criminal record"},
    {"domain": "Immigration", "policy": "Restricting work authorization for asylum seekers and undocumented immigrants"},
    {"domain": "Immigration", "policy": "Reducing refugee admissions and resettlement programs"},
    {"domain": "Immigration", "policy": "Ending or rolling back DACA-style protections"},
    {"domain": "Immigration", "policy": "Expanding state and local authority in immigration enforcement"},

    {"domain": "Healthcare", "policy": "Opposing Medicaid expansion and emphasizing state flexibility"},
    {"domain": "Healthcare", "policy": "Limiting Medicare drug price negotiations to preserve market competition"},
    {"domain": "Healthcare", "policy": "Reducing federal mandates on mental health insurance coverage"},
    {"domain": "Healthcare", "policy": "Restricting federal public health emergency powers"},
    {"domain": "Healthcare", "policy": "Reducing federal regulation of hospitals and insurance markets"},

    {"domain": "Economy", "policy": "Opposing increases to the federal minimum wage"},
    {"domain": "Economy", "policy": "Reducing or eliminating federal cash benefit programs"},
    {"domain": "Economy", "policy": "Opposing broad student loan forgiveness"},
    {"domain": "Economy", "policy": "Reducing zoning regulations primarily through local control"},
    {"domain": "Economy", "policy": "Limiting antitrust enforcement to avoid government overreach"},
    {"domain": "Economy", "policy": "Lowering taxes on high-income households and corporations"},

    {"domain": "Climate", "policy": "Rolling back emissions standards for vehicles and power plants"},
    {"domain": "Climate", "policy": "Reducing federal clean energy subsidies and tax credits"},
    {"domain": "Climate", "policy": "Fast-tracking permits for fossil fuel and energy infrastructure projects"},
    {"domain": "Climate", "policy": "Expanding oil and gas drilling on federal lands"},
    {"domain": "Climate", "policy": "Limiting environmental justice requirements in permitting decisions"},
    {"domain": "Climate", "policy": "Reducing federal water quality regulations in favor of state control"},

    {"domain": "Gun", "policy": "Opposing universal background check expansions"},
    {"domain": "Gun", "policy": "Opposing bans on assault weapons and high-capacity magazines"},
    {"domain": "Gun", "policy": "Expanding national concealed-carry reciprocity"},
    {"domain": "Gun", "policy": "Opposing or limiting red flag laws"},
    {"domain": "Gun", "policy": "Opposing federal police reform mandates"},
    {"domain": "Gun", "policy": "Opposing bail reform and emphasizing tough-on-crime policies"},
]


STANCE_BUCKETS = {
    "strongly object": -1,
    "object": -0.5,
    "neutral": 0,
    "support": 0.5,
    "strongly support": 1,
}


# ----------------------------
# Prompt builder
# ----------------------------

def build_prompt(policy: str, statement: str) -> str:
    return (
        f"USER: Generate a statement about the following policy: {policy}\n"
        f"ASSISTANT: {statement} I "
    )


# ----------------------------
# Main
# ----------------------------

rows = []

df_stances = pd.read_csv("../data/stances.csv")

for item in POLICIES:
    for i, row in df_stances.iterrows():
        stance = row["stance"]
        statement = row["statement"]
        label = STANCE_BUCKETS[stance]

        prompt = build_prompt(
            policy=item["policy"],
            statement=statement,
        )
        rows.append({
            "domain": item["domain"],
            "policy": item["policy"],
            "stance": stance,
            "statement": statement,
            "prompt": prompt,
            "label": label,
        })

df = pd.DataFrame(rows)
df.to_csv("../data/policy_statements_synthetic_stance.csv", index=False)

print(f"Saved {len(df)} prompts to policy_statements_synthetic_stance.csv")
