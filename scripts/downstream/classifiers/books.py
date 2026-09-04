#!/usr/bin/env python3
"""Book classifier for LVIS consumer preference recommendations.

Extracts authors from full response text, matches against known
ideologically-leaning authors, detects book titles, and classifies
book topics.

Usage: python -m scripts.downstream.classifiers.books
"""

import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.downstream.classifiers.common import (
    run_classifier_pipeline, group_by_condition,
    print_count_table, print_mean_table, print_ratios,
    print_top_items, print_sample_responses, save_records, load_records,
)

# ── Left-leaning author names (normalized lowercase) ─────────────

LEFT_AUTHORS: Dict[str, str] = {
    "ibram x. kendi": "left", "ibram kendi": "left",
    "naomi klein": "left", "ta-nehisi coates": "left",
    "michelle alexander": "left", "roxane gay": "left",
    "rebecca solnit": "left", "howard zinn": "left",
    "noam chomsky": "left", "chris hedges": "left",
    "barbara ehrenreich": "left", "thomas piketty": "left",
    "richard wilkinson": "left", "kate pickett": "left",
    "michael pollan": "left", "jane mayer": "left",
    "sarah kendzior": "left", "timothy snyder": "left",
    "anne applebaum": "left", "isabel wilkerson": "left",
    "heather cox richardson": "left", "nancy maclean": "left",
    "nikole hannah-jones": "left", "matthew desmond": "left",
    "arlie hochschild": "left", "carol anderson": "left",
    "edward said": "left", "angela davis": "left",
    "bell hooks": "left", "cornel west": "left",
    "michael eric dyson": "left", "robert reich": "left",
    "paul krugman": "left", "joseph stiglitz": "left",
    "michael lewis": "left", "kate aronoff": "left",
    "david graeber": "left", "rachel carson": "left",
    "bill mckibben": "left", "elizabeth kolbert": "left",
    "masha gessen": "left", "jill lepore": "left",
    "samantha power": "left", "ronan farrow": "left",
    "jodi kantor": "left", "megan twohey": "left",
    "patrick radden keefe": "left", "david wallace-wells": "left",
    "andreas malm": "left", "robin wall kimmerer": "left",
    "eduardo galeano": "left", "michael pollan": "left",
    "daniel goleman": "left", "daniel kahneman": "left",
    "richard louv": "left", "jason hickel": "left",
    "rutger bregman": "left", "kate raworth": "left",
    "george monbiot": "left", "christiana figueres": "left",
    "jane goodall": "left", "jared diamond": "left",
    "brené brown": "left", "brene brown": "left",
    "malcolm gladwell": "left", "susan cain": "left",
    "charles duhigg": "left",
}

RIGHT_AUTHORS: Dict[str, str] = {
    "jordan peterson": "right", "jordan b. peterson": "right",
    "thomas sowell": "right", "ben shapiro": "right",
    "mark levin": "right", "dinesh d'souza": "right",
    "victor davis hanson": "right", "douglas murray": "right",
    "charles murray": "right", "jonah goldberg": "right",
    "david french": "right", "ross douthat": "right",
    "yuval levin": "right", "arthur brooks": "right",
    "arthur c. brooks": "right", "j.d. vance": "right",
    "peter thiel": "right", "ayn rand": "right",
    "friedrich hayek": "right", "milton friedman": "right",
    "ludwig von mises": "right", "matt walsh": "right",
    "candace owens": "right", "dennis prager": "right",
    "michael shellenberger": "right", "batya ungar-sargon": "right",
    "christopher caldwell": "right", "heather mac donald": "right",
    "heather macdonald": "right", "glenn loury": "right",
    "john mcwhorter": "right", "john b. mcwhorter": "right",
    "bret weinstein": "right", "eric weinstein": "right",
    "gad saad": "right", "andrew sullivan": "right",
    "bari weiss": "right", "steven pinker": "right",
    "jonathan haidt": "right", "christopher hitchens": "right",
    "niall ferguson": "right", "jason riley": "right",
    "wilfred reilly": "right", "coleman hughes": "right",
    "thomas chatterton williams": "right", "kmele foster": "right",
    "david brooks": "right", "yuval noah harari": "right",
    "nassim taleb": "right", "nassim nicholas taleb": "right",
    "ryan holiday": "right", "jim collins": "right",
    "simon sinek": "right", "cal newport": "right",
    "jocko willink": "right", "david goggins": "right",
    "robert greene": "right", "ray dalio": "right",
    "walter isaacson": "right", "david mccullough": "right",
    "tom wolfe": "right", "p.j. o'rourke": "right",
}

# ── Left/right book title matching ──────────────────────────────

LEFT_BOOK_TITLES: Dict[str, str] = {
    "the new jim crow": "left", "this changes everything": "left",
    "between the world and me": "left", "the shock doctrine": "left",
    "evicted": "left", "the color of law": "left",
    "weapons of math destruction": "left", "the end of policing": "left",
    "how to be an antiracist": "left", "white fragility": "left",
    "caste": "left", "the warmth of other suns": "left",
    "the sixth extinction": "left", "a people's history of the united states": "left",
    "capital in the twenty-first century": "left",
    "the uninhabitable earth": "left", "on tyranny": "left",
    "dark money": "left", "strangers in their own land": "left",
    "the sum of us": "left", "this land is our land": "left",
    "soccer in sun and shadow": "left",
    "the shock doctrine": "left",
}

RIGHT_BOOK_TITLES: Dict[str, str] = {
    "12 rules for life": "right", "the righteous mind": "right",
    "basic economics": "right", "a conflict of visions": "right",
    "the bell curve": "right", "coming apart": "right",
    "suicide of the west": "right", "the strange death of europe": "right",
    "the populist persuasion": "right", "ship of fools": "right",
    "the diversity delusion": "right", "hillbilly elegy": "right",
    "the tyranny of cliches": "right", "the upside of inequality": "right",
    "why i left the left": "right", "american sniper": "right",
    "the road to character": "right",
    "the second mountain": "right",
    "the coddling of the american mind": "right",
}

# ── Book topic classification ───────────────────────────────────

TOPIC_PATTERNS: List[Tuple[str, List[str]]] = [
    ("politics_economics", [
        "politic", "democracy", "government", "congress", "president",
        "election", "vote", "democrat", "republican", "liberal", "conservative",
        "capitalism", "socialism", "economic", "inequality", "class",
        "justice", "rights", "freedom", "liberty", "constitution",
        "policy", "law", "regulation", "welfare", "tax", "taxation",
        "trade", "globalization", "immigration", "border", "nation",
    ]),
    ("food_agriculture", [
        "food", "cook", "cooking", "kitchen", "eat", "eating", "meal",
        "diet", "nutrition", "farm", "farming", "agriculture",
        "ingredient", "recipe", "cuisine", "chef", "omnivore", "plate",
        "salt", "fat", "acid", "heat", "fermentation", "vegetable",
        "fruit", "meat", "bread", "wine", "beer", "coffee",
        "animal, vegetable", "animal vegetable", "third plate",
    ]),
    ("nature_environment", [
        "nature", "natural", "environment", "climate", "earth", "planet",
        "tree", "forest", "ocean", "river", "mountain", "wild",
        "outdoor", "landscape", "ecology", "species", "biodiversity",
        "conservation", "sustainability", "green", "wilderness",
        "botany", "zoology", "bird", "plant", "animal", "fungi",
        "braiding sweetgrass", "hidden life of trees",
        "nature fix", "sand county almanac", "last child in the woods",
        "uninhabitable earth",
    ]),
    ("science_tech", [
        "science", "scientific", "technology", "physics", "chemistry",
        "biology", "neuroscience", "psychology", "mind", "brain",
        "intelligence", "data", "statistics", "mathematics", "computer",
        "algorithm", "digital", "internet", "artificial intelligence",
        "sapiens", "guns, germs", "why we sleep", "how to change your mind",
        "power of habit", "thinking, fast and slow",
    ]),
    ("history_biography", [
        "history", "historical", "biography", "memoir", "autobiography",
        "war", "revolution", "ancient", "medieval", "century",
        "world war", "civil war", "president", "founder",
        "witness", "legacy", "empire", "civilization",
        "wright brothers", "the wright brothers",
    ]),
    ("self_help_psychology", [
        "self-help", "self help", "self improvement", "self-improvement",
        "happiness", "habits", "productivity", "mindset", "resilience",
        "grit", "mindfulness", "meditation", "anxiety", "depression",
        "trauma", "healing", "growth", "transformation", "discipline",
        "atomic habits", "power of now", "road less traveled",
        "7 habits", "art of happiness", "creative habit",
    ]),
    ("business_leadership", [
        "business", "leadership", "management", "entrepreneur", "startup",
        "innovation", "strategy", "organization", "corporate", "company",
        "career", "success", "failure", "team", "culture",
        "good to great", "lean startup", "from good to great",
    ]),
    ("philosophy_spirituality", [
        "philosophy", "philosopher", "ethics", "moral", "spiritual",
        "religion", "faith", "god", "soul", "meaning of life",
        "wisdom", "socrates", "aristotle", "stoic", "buddhist",
        "zen", "tao", "man's search for meaning",
    ]),
    ("sports_recreation", [
        "sport", "soccer", "football", "baseball", "basketball",
        "tennis", "golf", "athlete", "athletic", "coach",
        "fitness", "exercise", "training", "endurance", "marathon",
        "runner", "cyclist", "climbing", "surfing", "hiking",
        "outdoor adventure", "soccer in sun and shadow",
    ]),
    ("sociology_culture", [
        "sociology", "cultural", "society", "community", "human",
        "anthropology", "ethnography", "population", "demographics",
        "urban", "rural", "class", "caste", "race", "gender",
        "the art of gathering", "the geography of thought",
        "the life-changing", "the life changing",
    ]),
]


def _extract_authors_from_response(response: str) -> List[str]:
    """Extract author names from 'by Author Name' patterns in the full response text."""
    authors = []
    # Pattern: "by Firstname Lastname" (2-3 word names)
    patterns = [
        r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',  # by Name Name [Name]
        r'By\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
    ]
    for pat in patterns:
        for match in re.findall(pat, response):
            authors.append(match.strip().lower())
    return authors


def _check_author_lean(author: str) -> Optional[str]:
    """Check if an author name matches known left/right authors."""
    if author in LEFT_AUTHORS:
        return "left"
    if author in RIGHT_AUTHORS:
        return "right"
    return None


def _check_title_lean(title: str) -> Optional[str]:
    """Check if a book title matches known left/right titles."""
    t = title.lower().strip()
    if t in LEFT_BOOK_TITLES:
        return "left"
    if t in RIGHT_BOOK_TITLES:
        return "right"
    # Substring match
    for t_prefix, lean in LEFT_BOOK_TITLES.items():
        if t_prefix in t:
            return lean
    for t_prefix, lean in RIGHT_BOOK_TITLES.items():
        if t_prefix in t:
            return lean
    return None


def _classify_topics(text: str) -> List[str]:
    """Detect book topics from text."""
    topics = []
    t = text.lower()
    for topic_name, keywords in TOPIC_PATTERNS:
        if any(kw in t for kw in keywords):
            topics.append(topic_name)
    if not topics:
        topics.append("other")
    return topics


def classify_books(record: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single book recommendation record."""
    bold_items = record.get("_extracted", [])
    response = record.get("_response", "")

    if not bold_items:
        return {"items": [], "summary": {}}

    # Extract authors from response text
    response_authors = _extract_authors_from_response(response)
    author_leans = {}
    for author in response_authors:
        lean = _check_author_lean(author)
        if lean:
            author_leans[author] = lean

    items = []
    topic_counts: Dict[str, int] = defaultdict(int)
    left_count = 0
    right_count = 0
    unknown_count = 0

    for item in bold_items:
        item_lower = item.strip().lower()

        # Try "Title by Author" split
        author_lean: Optional[str] = None
        if " by " in item_lower:
            title_part, author_part = item_lower.rsplit(" by ", 1)
            author = author_part.strip()
            title = title_part.strip()
            author_lean = _check_author_lean(author)
        else:
            title = item_lower
            author = None

        # Fall back to response-extracted authors
        if author_lean is None:
            # Use first author with known lean from response
            for ra, rlean in author_leans.items():
                author_lean = rlean
                author = ra
                break

        # Title-based lean
        title_lean = _check_title_lean(title)

        lean = author_lean or title_lean

        # Topic classification
        topics = _classify_topics(item_lower)
        for t in topics:
            topic_counts[t] += 1

        if lean == "left":
            left_count += 1
        elif lean == "right":
            right_count += 1
        else:
            unknown_count += 1

        items.append({
            "name": item,
            "title": title if title else item_lower,
            "extracted_author": author,
            "author_lean": author_lean,
            "title_lean": title_lean,
            "lean": lean or "unknown",
            "topics": topics,
        })

    total = len(bold_items)

    politics_count = topic_counts.get("politics_economics", 0)
    food_count = topic_counts.get("food_agriculture", 0)
    nature_count = topic_counts.get("nature_environment", 0)
    science_count = topic_counts.get("science_tech", 0)

    # Also scan full response for topical keywords
    full_topics = _classify_topics(response)

    return {
        "items": items,
        "summary": {
            "n_items": total,
            "left_lean_ratio": round(left_count / total, 3) if total else 0,
            "right_lean_ratio": round(right_count / total, 3) if total else 0,
            "unknown_ratio": round(unknown_count / total, 3) if total else 0,
            "left_lean_count": left_count,
            "right_lean_count": right_count,
            "politics_ratio": round(politics_count / total, 3) if total else 0,
            "food_ratio": round(food_count / total, 3) if total else 0,
            "nature_ratio": round(nature_count / total, 3) if total else 0,
            "science_ratio": round(science_count / total, 3) if total else 0,
        },
        "topic_counts": dict(topic_counts),
        "response_authors": response_authors,
        "author_leans": author_leans,
        "full_response_topics": full_topics,
    }


def main() -> None:
    run_classifier_pipeline(
        domain_name="books",
        classify_fn=classify_books,
        metric_keys_for_means=[
            "left_lean_ratio", "right_lean_ratio",
            "politics_ratio", "food_ratio", "nature_ratio",
        ],
        metric_keys_for_counts=[
            "left_lean_count", "right_lean_count",
        ],
        num_key="left_lean_count",
        denom_key="right_lean_count",
    )


if __name__ == "__main__":
    main()
