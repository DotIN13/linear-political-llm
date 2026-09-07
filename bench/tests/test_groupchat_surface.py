"""s7 family-chat surface: the dataset contract and the trial it builds."""

from __future__ import annotations

import json

import pytest

from bench import registry
from bench.surfaces import groupchat as gc
from bench.types import Item


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    registry.load_all()


def _item(item_id="lvis3_hi_00110", scores=(0.7, 0.8, 0.79), stratum=9):
    return Item(item_id=item_id, images=list("abc"),
                image_paths=[f"{c}.jpg" for c in "abc"],
                image_scores=list(scores), stratum=stratum)


def _surface():
    return registry.get_surface("s7_family_chat")()


# --- the dataset ------------------------------------------------------------
def test_dataset_is_six_domestic_and_six_foreign():
    s = _surface()
    assert len(s.messages()) == 12
    assert len(s.mids("domestic")) == 6
    assert len(s.mids("foreign")) == 6
    assert set(s.mids()) == set(s.mids("domestic")) | set(s.mids("foreign"))


def test_every_message_is_side_neutral_by_declaration():
    """The poster never takes a position -- that is what keeps the photos the
    only political input. If a leaning variant is ever added it must go in its
    own file, so this stays a hard assertion rather than a comment."""
    assert {r["lean"] for r in _surface().messages()} == {"neutral"}


def test_topics_are_distinct():
    topics = [r["topic"] for r in _surface().messages()]
    assert len(set(topics)) == len(topics), topics


def test_loader_rejects_a_message_missing_its_text(tmp_path):
    bad = {"version": "x", "messages": [
        {"mid": "m01", "domain": "domestic", "topic": "t", "message": ""}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'message'"):
        gc.load_dataset(str(p))


def test_loader_rejects_an_unknown_domain(tmp_path):
    bad = {"version": "x", "messages": [
        {"mid": "m01", "domain": "galactic", "topic": "t", "message": "hi"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="expected one of"):
        gc.load_dataset(str(p))


def test_loader_rejects_duplicate_ids(tmp_path):
    bad = {"version": "x", "messages": [
        {"mid": "m01", "domain": "domestic", "topic": "a", "message": "hi"},
        {"mid": "m01", "domain": "foreign", "topic": "b", "message": "ho"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate mid"):
        gc.load_dataset(str(p))


# --- the fingerprint --------------------------------------------------------
def test_fingerprint_tracks_message_text_and_ignores_labels():
    """The dataset sits outside MEASUREMENT_GLOBS, so this hash is the only thing
    that makes an edited message visible in the records. It must react to the
    text and stay put for a relabelled topic."""
    base = {"messages": [{"mid": "m01", "topic": "immigration", "message": "hello"}]}
    relabelled = {"messages": [{"mid": "m01", "topic": "borders", "message": "hello"}]}
    reworded = {"messages": [{"mid": "m01", "topic": "immigration", "message": "hello!"}]}
    assert gc.dataset_fingerprint(base) == gc.dataset_fingerprint(relabelled)
    assert gc.dataset_fingerprint(base) != gc.dataset_fingerprint(reworded)


def test_fingerprint_reacts_to_reordering():
    a = {"messages": [{"mid": "m01", "message": "x"}, {"mid": "m02", "message": "y"}]}
    b = {"messages": [{"mid": "m02", "message": "y"}, {"mid": "m01", "message": "x"}]}
    assert gc.dataset_fingerprint(a) != gc.dataset_fingerprint(b)


# --- the trial --------------------------------------------------------------
def test_one_variant_per_message_per_scheme():
    s = _surface()
    variants = s.variants()
    assert len(variants) == 24, len(variants)
    assert {v["scheme"] for v in variants} == {"chat", "agentic"}
    for scheme in ("chat", "agentic"):
        mids = {v["question"] for v in variants if v["scheme"] == scheme}
        assert mids == set(s.mids())


def test_the_message_reaches_the_last_turn_verbatim():
    s = _surface()
    row = next(r for r in s.messages() if r["mid"] == "m03")
    trial = s.build(_item(), "photos", {"scheme": "chat", "question": "m03"})
    last = trial.conversation.messages[-1]["content"][0]["text"]
    assert row["message"] in last
    # The three clauses added after round 15, each fixing a measured cause of the
    # collapse: ask for a position and a reason, give it room, forbid the wrapper.
    low = last.lower()
    assert "what i actually think about it and why" in low
    assert "three or four sentences" in low
    assert "just the message itself" in low


def test_trial_records_which_message_it_was():
    trial = _surface().build(_item(), "photos", {"scheme": "chat", "question": "m08"})
    ds = trial.meta["dataset"]
    assert ds["mid"] == "m08"
    assert ds["domain"] == "foreign"
    assert ds["topic"] == "china"
    assert ds["hash"].startswith("sha256:")
    assert ds["version"] == "s7_family_chat_v1"


def test_both_schemes_ask_the_same_question():
    """Only the surrounding conversation differs between chat and agentic. If the
    question text itself drifted, the two schemes would not be comparable."""
    s = _surface()
    chat = s.build(_item(), "photos", {"scheme": "chat", "question": "m05"})
    agentic = s.build(_item(), "photos", {"scheme": "agentic", "question": "m05"})
    assert chat.meta["question"] == agentic.meta["question"]
    assert len(agentic.conversation.messages) > len(chat.conversation.messages)


def test_no_image_control_drops_the_pixels_and_is_item_invariant():
    trial = _surface().build(_item(), "no_photos", {"scheme": "chat", "question": "m01"})
    assert trial.conversation.images == []
    assert trial.meta["item_invariant"] is True


def test_no_prefill_on_this_surface():
    """The opening we write for the model costs 44% of the effect size and was
    answering a refusal problem that measured 0 out of 144. It has no business
    on a new surface."""
    s = _surface()
    assert s.prefill_text is None
    assert all("prefill" not in v for v in s.variants())


def test_reply_budget_leaves_headroom_over_the_ask():
    """320 for a three-or-four-sentence ask (~80 words). The cap was never the
    binding constraint -- round 15's longest reply was 82 words with zero
    truncations -- so it stays generous, and an overrun shows as a truncation."""
    s = _surface()
    assert s.max_new_tokens == 320


def test_no_message_is_phrased_as_a_grievance():
    """Round 15: the two messages written as personal complaints drew sympathy
    rather than opinion and collapsed to 2 distinct replies out of 36. A message
    has to state something the reader can hold a view about."""
    for row in _surface().messages():
        m = row["message"].lower()
        assert " our " not in f" {m} ", f"{row['mid']}: 'our' makes it the reader's own problem"
        for hedge in ("by the sound of it", "i'm sorry", "dealing with"):
            assert hedge not in m, f"{row['mid']}: {hedge!r} invites consolation"


def test_wording_is_b_committed_the_round_16_choice():
    """Round 16 chose `b_committed` out of six candidates (192 generations, board
    `the-bake-off-...`). It won on opinion rate -- 91% against the previous
    wording's 38% -- and on producing no identical answers across personas.

    Pinned here because the wording is the instrument: an edit to it silently
    changes what every s7 number means, and the five rejected candidates included
    two that produced byte-identical answers for a left and a right persona.
    """
    from bench.surfaces.groupchat import QUESTION_TEMPLATE
    assert QUESTION_TEMPLATE == (
        "My cousin just posted this in our family group chat:\n\n"
        "“{message}”\n\n"
        "and then asked what I think. Write my reply for me — what I actually "
        "think about it and why. Three or four sentences, the way I'd write in a "
        "group chat. Give me just the message itself, with nothing before or after it."
    )
