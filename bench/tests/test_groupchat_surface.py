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
    trial = s.build(_item(), "C", {"scheme": "chat", "question": "m03"})
    last = trial.conversation.messages[-1]["content"][0]["text"]
    assert row["message"] in last
    assert "one or two" in last.lower()


def test_trial_records_which_message_it_was():
    trial = _surface().build(_item(), "C", {"scheme": "chat", "question": "m08"})
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
    chat = s.build(_item(), "C", {"scheme": "chat", "question": "m05"})
    agentic = s.build(_item(), "C", {"scheme": "agentic", "question": "m05"})
    assert chat.meta["question"] == agentic.meta["question"]
    assert len(agentic.conversation.messages) > len(chat.conversation.messages)


def test_no_image_control_drops_the_pixels_and_is_item_invariant():
    trial = _surface().build(_item(), "E", {"scheme": "chat", "question": "m01"})
    assert trial.conversation.images == []
    assert trial.meta["item_invariant"] is True


def test_no_prefill_on_this_surface():
    """The opening we write for the model costs 44% of the effect size and was
    answering a refusal problem that measured 0 out of 144. It has no business
    on a new surface."""
    s = _surface()
    assert s.prefill_text is None
    assert all("prefill" not in v for v in s.variants())


def test_reply_budget_is_short_but_not_so_short_it_hides_an_essay():
    s = _surface()
    assert s.max_new_tokens == 200
