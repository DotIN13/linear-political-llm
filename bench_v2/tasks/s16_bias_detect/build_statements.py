"""Build ``statements_v1.jsonl``: eighteen statements from the paper's 240.

The pool is the political-statements dataset of Zhang (Findings of EMNLP 2025,
"Probing Political Ideology in Large Language Models"), ``data/policy_statements_240.csv``:
one gpt-4o-mini statement per (116th-Congress lawmaker, policy domain), 40
lawmakers x 6 domains. Every trial runs every statement, so the full 240 would
multiply the run by twenty.

Three per domain, one of each **text** lean: left, right and balanced. The lean is
the text's, not the author's -- gpt-4o-mini wrote many conservative lawmakers a
centrist sentence, so the author's DW-NOMINATE score alone does not give a right
text. The 240 carry no text label, so the three were chosen by reading every
statement in the domain, under a rule stated before choosing:

* **left**: the most liberal author (lowest ``nominate_dim1``) whose text plainly
  takes the left side;
* **right**: the most conservative author whose text plainly takes the right side;
* **balanced**: the author nearest zero whose text weighs both sides or takes none.

That reading is a judgement (made by Claude, 2026-09-28) and is recorded per row in
``text_lean`` and ``why``. The s16 no-photos baseline is the model's own reading of
the same eighteen, and the s17 judge can be pointed at the originals as a check.

Ukraine is the weak domain: all 40 statements support the aid in nearly the same
words, so its three differ only in shading (a call for oversight, a generic
endorsement). They are kept so every domain has three, and flagged in ``why``.

Both s16_bias_detect and s17_neutralize read this one file, so the two tasks are
a contrast on identical statements (classify it vs rewrite it neutrally).

    python -m bench_v2.tasks.s16_bias_detect.build_statements
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STATEMENTS = ROOT / "data" / "policy_statements_240.csv"
MEMBERS = ROOT / "data" / "HS116_members.csv"
OUT = HERE / "statements_v1.jsonl"
DOMAINS = ("Abortion", "Border Wall", "Transgender Bathroom",
           "Ukraine Military Aid", "Israel Military Aid", "Gaza Humanitarian Aid")


def fullname(bioname: str) -> str:
    """``"LEAHY, Patrick Joseph"`` -> ``"Patrick J. Leahy"``, as create_statements.py does."""
    last, first_middle = bioname.split(", ")[0], bioname.split(", ")[1]
    parts = first_middle.split(" ")
    middle = parts[1][0] + ". " if len(parts) > 1 else ""
    return parts[0] + " " + middle + last.lower().capitalize().strip()


def clean(text: str) -> str:
    """The csv wraps some statements in a stray pair of quotes; drop them."""
    text = text.strip()
    while len(text) > 1 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


# (domain, text lean) -> (author, why). Chosen by reading all 40 texts in the domain.
PICKS: dict[tuple[str, str], tuple[str, str]] = {
    ("Abortion", "left"): ("John R. Lewis", "a fundamental right that must be protected; most liberal author"),
    ("Abortion", "right"): ("Benjamin Cline", "protect the rights of the unborn; Andrew Biggs (+0.83) is more conservative but his text calls for a balanced approach"),
    ("Abortion", "balanced"): ("Adam Kinzinger", "compassion, individual rights and women's health, no side taken; Susan Collins (+0.12) is nearer zero but her text leans pro-choice"),
    ("Border Wall", "left"): ("Janice D. Schakowsky", "humane solutions over a wall that divides communities; most liberal author"),
    ("Border Wall", "right"): ("Michael Cloud", "a wall is essential to safety and sovereignty; most conservative author"),
    ("Border Wall", "balanced"): ("Christopher A. Coons", "both enforcement and humane solutions, not only a wall; Mark Warner (-0.21) is nearer zero but his text opposes the wall"),
    ("Transgender Bathroom", "left"): ("Sylvia Garcia", "access that aligns with gender identity; most liberal author"),
    ("Transgender Bathroom", "right"): ("James M. Inhofe", "facilities by biological sex; the only plainly right text in the domain"),
    ("Transgender Bathroom", "balanced"): ("Joe Manchin", "a balanced approach, dignity and safety; nearest zero"),
    ("Ukraine Military Aid", "left"): ("Kamala D. Harris", "WEAK DOMAIN: generic support for aid, as all 40 are; most liberal author"),
    ("Ukraine Military Aid", "right"): ("Gary J. Palmer", "WEAK DOMAIN: support, but targeted aid with oversight, the only right-coded shading in the domain"),
    ("Ukraine Military Aid", "balanced"): ("Collin C. Peterson", "WEAK DOMAIN: support framed as regional stability; nearest zero"),
    ("Israel Military Aid", "left"): ("Elizabeth Warren", "aid conditioned on protecting Palestinian civilians; most liberal author"),
    ("Israel Military Aid", "right"): ("William Steube", "continued aid is essential for an ally; most conservative author"),
    ("Israel Military Aid", "balanced"): ("Thomas W. Reed", "aid that brings peace and security for both Israelis and Palestinians; nearest zero that weighs both"),
    ("Gaza Humanitarian Aid", "left"): ("Elizabeth Warren", "immediate aid and the rights of all people; most liberal author"),
    ("Gaza Humanitarian Aid", "right"): ("Tom Mcclintock", "aid must not support terrorist organizations; Lance Gooden (+0.75) is more conservative but his text takes no side"),
    ("Gaza Humanitarian Aid", "balanced"): ("Gordon D. Jones", "aid for civilians, no side taken; nearest zero"),
}


def check_rule(domain: str, rows: list[dict[str, str]],
               nominate: dict[str, tuple[float, str]]) -> None:
    """A cheap guard on the hand picks: the left text's author must score below zero
    and the right text's author above it. The rest of the rule is in ``why``."""
    left = nominate[PICKS[(domain, "left")][0]][0]
    right = nominate[PICKS[(domain, "right")][0]][0]
    if not (left < 0 < right):
        raise ValueError(f"{domain}: left author {left:+.2f} / right author {right:+.2f}")


def main() -> int:
    nominate: dict[str, tuple[float, str]] = {}
    with MEMBERS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["nominate_dim1"]:
                nominate.setdefault(fullname(row["bioname"]),
                                    (float(row["nominate_dim1"]), row["party_code"]))
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    with STATEMENTS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["name"] in nominate:
                by_domain[row["domain"]].append(row)

    rows = []
    for d, domain in enumerate(DOMAINS, start=1):
        by_name = {row["name"]: row for row in by_domain[domain]}
        for lean in ("left", "balanced", "right"):
            name, why = PICKS[(domain, lean)]
            row = by_name[name]
            dim1, party = nominate[name]
            rows.append({
                "sid": f"st{d:02d}{lean[0]}",
                "domain": domain,
                "text_lean": lean,
                "author": name,
                "author_party": {"100": "D", "200": "R"}.get(party, party),
                "author_nominate_dim1": dim1,
                "statement": clean(row["response"]),
                "why": why,
            })
        check_rule(domain, by_domain[domain], nominate)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    blob = STATEMENTS.read_bytes()
    meta = {
        "version": "statements_v1",
        "n_items": len(rows),
        "source": "data/policy_statements_240.csv (huggingface.co/datasets/DotIN13/political-statements)",
        "source_sha256": hashlib.sha256(blob).hexdigest()[:16],
        "paper": "Zhang, Probing Political Ideology in Large Language Models, Findings of EMNLP 2025, 23349-23360",
        "rule": "per domain one text of each lean: left = most liberal author whose text plainly takes the left side; "
                "right = most conservative author whose text plainly takes the right side; balanced = author nearest "
                "zero whose text weighs both sides or takes none. Six domains x three = 18",
        "labelled_by": "hand reading by Claude, 2026-09-28; the reason for each pick is in `why`",
        "note": "Ukraine is weak: all 40 statements support the aid in nearly the same words, so its three differ "
                "only in shading.",
        "built_by": "bench_v2/tasks/s16_bias_detect/build_statements.py",
    }
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} statements to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
