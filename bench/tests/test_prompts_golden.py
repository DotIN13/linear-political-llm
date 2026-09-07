"""The characterisation test. If this goes red, a prompt changed -- find out why.

``bench/surfaces/generation.py`` (1,143 lines, six unrelated questions plus all the
shared machinery) was split into ``bench/surfaces/shared/`` and
``bench/surfaces/questions/``. The claim being made is that **not one byte of any
built prompt changed.** The other tests in this suite would not catch a changed
prompt; this one is the whole warrant.

It is not a unit test and it is not asserting that any particular prompt is *right*.
It asserts only that the prompts are what they were at the commit that added the
golden files -- which is what a refactor is allowed to do to them, and nothing else.

Two snapshots:

* ``golden/prompts.jsonl`` -- the full cross product of surface x scheme/question x
  condition x item x order, rebuilt and compared as canonical JSON.
* ``golden/readers.jsonl`` -- the deterministic readers (``word_count``,
  ``detect_refusal``, ``_refusal_match``, ``extract_topic``,
  ``extract_mentions_politics``, ``extract_picks``) over ``golden/responses.jsonl``,
  which is real model output from four runs on midway3 plus a labelled synthetic
  tail that pins the eight refusal patterns no run has ever produced.

**When a prompt is meant to change**, regenerate with
``python -m bench.tests.prompt_enumeration --write``, and expect the diff to be
reviewed -- and note that it invalidates ``measurement_rev``, so the affected trials
have to be re-run.
"""

import json

import pytest

from bench.tests.prompt_enumeration import (
    PROMPTS_PATH, READERS_PATH, RESPONSES_PATH, canonical, enumerate_all_prompts,
    enumerate_all_readers,
)


def _golden(path):
    """The snapshot as {key: canonical line}. Duplicate keys are a bug in the enumerator."""
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            key = json.loads(line)["key" if "prompts" in path else "id"]
            assert key not in rows, f"{path}:{lineno} duplicate key {key!r}"
            rows[key] = line
    return rows


GOLDEN_PROMPTS = _golden(PROMPTS_PATH)
GOLDEN_READERS = _golden(READERS_PATH)


def test_the_golden_files_are_not_empty():
    """Guards the vacuous pass: an emptied snapshot must not look like agreement."""
    assert len(GOLDEN_PROMPTS) >= 1000
    assert len(GOLDEN_READERS) >= 1800
    with open(RESPONSES_PATH, encoding="utf-8") as handle:
        assert sum(1 for line in handle if line.strip()) == len(GOLDEN_READERS)


def test_every_prompt_is_unchanged():
    """Byte for byte, not 'equivalent'."""
    seen = set()
    for row in enumerate_all_prompts():
        key = row["key"]
        seen.add(key)
        assert key in GOLDEN_PROMPTS, f"prompt not in the golden file: {key}"
        assert canonical(row) == GOLDEN_PROMPTS[key], f"prompt changed: {key}"
    missing = sorted(set(GOLDEN_PROMPTS) - seen)
    assert not missing, f"{len(missing)} prompts in the golden file are no longer built: {missing[:3]}"


def test_every_reader_is_unchanged():
    seen = set()
    for row in enumerate_all_readers():
        key = row["id"]
        seen.add(key)
        assert key in GOLDEN_READERS, f"response not in the golden file: {key}"
        assert canonical(row) == GOLDEN_READERS[key], f"reader output changed: {key}"
    missing = sorted(set(GOLDEN_READERS) - seen)
    assert not missing, f"{len(missing)} responses no longer read: {missing[:3]}"


# --- what the snapshot is worth ----------------------------------------------
# A snapshot only proves what it covers, so the coverage is asserted rather than
# hoped for. These numbers are the argument that the refactor is verifiable.
def test_the_prompt_snapshot_covers_every_registered_surface():
    from bench import registry

    registry.load_all()
    covered = {json.loads(line)["trial"]["surface"] for line in GOLDEN_PROMPTS.values()}
    assert covered == set(registry.surface_names())


def test_the_reader_snapshot_covers_every_refusal_pattern():
    """All twelve branches of ``_refusal_match``, or the snapshot is not a snapshot."""
    import re

    from bench.surfaces.generation import (
        REFUSAL_WINDOW, _REFUSAL_PATTERNS, _normalize_apostrophes,
    )

    with open(RESPONSES_PATH, encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in handle if line.strip()]
    heads = [_normalize_apostrophes(t[:REFUSAL_WINDOW]).lower() for t in texts]
    unhit = [p for p in _REFUSAL_PATTERNS if not any(re.search(p, h) for h in heads)]
    assert not unhit, f"refusal patterns with no example in the corpus: {unhit}"


def test_the_reader_snapshot_covers_every_extract_picks_path():
    """Both parse_ok values against every match_method the matcher can report."""
    seen = set()
    for line in GOLDEN_READERS.values():
        picks = json.loads(line).get("extract_picks")
        if picks:
            seen.add((picks["parse_ok"], picks["match_method"]))
    for method in ("index", "outlet", "fuzzy", "mixed"):
        assert (True, method) in seen, f"no parse_ok row matched by {method}"
        assert (False, method) in seen, f"no failed row matched by {method}"
    assert (False, "none") in seen


@pytest.mark.parametrize("slug", [
    "climate", "healthcare", "housing", "education", "taxes", "crime", "border",
    "regulation", "other",
])
def test_the_reader_snapshot_covers_every_topic_slug(slug):
    """Every branch of ``extract_topic``, including the ``other`` fallthrough."""
    slugs = {json.loads(line)["extract_topic"]["topic_slug"] for line in GOLDEN_READERS.values()}
    assert slug in slugs
