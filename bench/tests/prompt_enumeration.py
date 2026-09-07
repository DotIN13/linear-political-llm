"""The characterisation ("golden") enumeration: every prompt the surfaces can build.

Why this exists. ``bench/surfaces/generation.py`` was split into
``bench/surfaces/shared/`` + ``bench/surfaces/questions/`` in one behaviour-preserving
refactor. A refactor of prompt-building code is only safe if the prompts can be
*proved* unchanged, and reading a diff does not prove it. So: enumerate the full cross
product of ``surface x scheme/question x condition x item x order``, serialise every
built trial canonically, and commit the result **before** moving a line. After the move
the same enumeration has to reproduce that file byte for byte.

It covers every *registered* surface, not only the six generation questions, because
``s7_family_chat`` and ``s8_letter_answered`` subclass ``GenerationSurface`` and the
multiple-choice surfaces share ``bench/surfaces/base.py``: a break in the shared
machinery would show up there first.

Regenerate deliberately (and only when a prompt is *meant* to change):

    python -m bench.tests.prompt_enumeration --write

The readers are snapshotted the same way, against **real model output** pulled from
``runs/factorial_three_tasks`` and ``runs/pilot_s3``: see ``golden/responses.jsonl``.
"""

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Dict, Iterator, List, Optional

from bench import registry
from bench.types import Item

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
PROMPTS_PATH = os.path.join(GOLDEN_DIR, "prompts.jsonl")
RESPONSES_PATH = os.path.join(GOLDEN_DIR, "responses.jsonl")
READERS_PATH = os.path.join(GOLDEN_DIR, "readers.jsonl")


def canonical(payload: Any) -> str:
    """The one serialisation. Same settings as ``bench.types._canonical_json``."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --- the stimulus fixtures ---------------------------------------------------
# Three photo counts, because the number of photos changes the transcript shape:
# 1 forces `files_by_dir` to keep both directories non-empty, 3 is the historical
# default, 10 is the largest the filename pool allows.
def _item(item_id: str, n: int, stratum: int) -> Item:
    return Item(
        item_id=item_id,
        images=[f"train2017/{i:012d}.jpg" for i in range(n)],
        image_paths=[f"/tmp/golden/{item_id}_{i}.jpg" for i in range(n)],
        image_scores=[round(-0.5 + 0.1 * i, 4) for i in range(n)],
        stratum=stratum,
        covariates={"n_objects": list(range(n))},
        split="explore",
    )


ITEMS: List[Item] = [
    _item("gold_one", 1, 0),
    _item("gold_three", 3, 5),
    _item("gold_ten", 10, 9),
]

SEEDS = (0, 42)

# s3's own axes. ``None`` means "let the surface deal its own order from the seed";
# the two pinned orders are what makes order an enumerable factor rather than a
# hidden draw. Both are 12 long, as a real deal is.
S3_ORDERS: List[Optional[List[int]]] = [
    None,
    list(range(12)),
    [11, 9, 7, 5, 3, 1, 10, 8, 6, 4, 2, 0],
]
S3_ATTRIBUTIONS = ("shown", "hidden")


def _accepts_seed(surface: Any) -> bool:
    return "seed" in inspect.signature(surface.build).parameters


def _trial_payload(trial: Any) -> Dict[str, Any]:
    """Everything about the built trial that is deterministic -- which is all of it."""
    return {
        "surface": trial.surface,
        "item_id": trial.item_id,
        "condition": trial.condition,
        "messages": trial.conversation.messages,
        "images": trial.conversation.images,
        "conversation_sha": trial.conversation.sha,
        "candidates": list(trial.candidates),
        "probe_points": [{"name": p.name, "kind": p.kind, "reduce": p.reduce}
                         for p in trial.probe_points],
        "max_new_tokens": trial.max_new_tokens,
        "variant": trial.variant,
        "meta": trial.meta,
    }


def _cases(surface_id: str) -> Iterator[Dict[str, Any]]:
    """(condition, variant, item, seed) for one surface, in a fixed order."""
    surface = registry.get_surface(surface_id)()
    generation = getattr(surface, "family", "") == "generation"
    seeds = SEEDS if _accepts_seed(surface) else (None,)
    for condition in surface.conditions:
        for variant in surface.variants():
            extras: List[Dict[str, Any]] = [{}]
            if generation and getattr(surface, "headlines", None):
                extras = [{"order": order, "attribution": attribution}
                          for order in S3_ORDERS for attribution in S3_ATTRIBUTIONS]
            for extra in extras:
                for item in ITEMS:
                    for seed in seeds:
                        full = dict(variant)
                        for key, value in extra.items():
                            if value is None:
                                continue        # let the surface deal its own
                            full[key] = value
                        yield {"condition": condition, "variant": full,
                               "item": item, "seed": seed,
                               "order_pinned": extra.get("order") is not None}


def enumerate_all_prompts() -> Iterator[Dict[str, Any]]:
    """Every prompt, keyed. Deterministic order: sorted surface, then the cross product."""
    registry.load_all()
    for surface_id in sorted(registry.surface_names()):
        surface = registry.get_surface(surface_id)()
        accepts_seed = _accepts_seed(surface)
        for case in _cases(surface_id):
            item = case["item"]
            key = canonical({
                "surface": surface_id,
                "condition": case["condition"],
                "variant": case["variant"],
                "item": item.item_id,
                "seed": case["seed"],
            })
            if accepts_seed:
                trial = surface.build(item, case["condition"], dict(case["variant"]),
                                      case["seed"])
            else:
                trial = surface.build(item, case["condition"], dict(case["variant"]))
            yield {"key": key, "trial": _trial_payload(trial)}


# --- the readers, against real model output ---------------------------------
def _reader_row(row: Dict[str, Any], headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
    from bench.surfaces.generation import (
        _refusal_match, detect_refusal, extract_mentions_politics, extract_picks,
        extract_topic, word_count,
    )

    text = row["text"]
    out: Dict[str, Any] = {
        "id": row["id"],
        "word_count": word_count(text),
        "detect_refusal": detect_refusal(text),
        "refusal_match": _refusal_match(text),
        "extract_topic": extract_topic(text),
        "extract_mentions_politics": extract_mentions_politics(text),
    }
    if row.get("order") is not None:
        out["extract_picks"] = extract_picks(text, headlines, row["order"])
    return out


def enumerate_all_readers() -> Iterator[Dict[str, Any]]:
    """Every deterministic reader, over every committed real response."""
    from bench.surfaces.generation import load_s3_headlines

    headlines = load_s3_headlines()
    with open(RESPONSES_PATH, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _reader_row(json.loads(line), headlines)


def _write(path: str, rows: Iterator[Dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
            n += 1
    return n


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="overwrite the golden files -- only when a change is intended")
    args = parser.parse_args()
    if not args.write:
        print(f"{sum(1 for _ in enumerate_all_prompts())} prompts, "
              f"{sum(1 for _ in enumerate_all_readers())} reader rows (nothing written)")
        return
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    print(f"prompts.jsonl: {_write(PROMPTS_PATH, enumerate_all_prompts())} rows")
    print(f"readers.jsonl: {_write(READERS_PATH, enumerate_all_readers())} rows")


if __name__ == "__main__":
    main()
