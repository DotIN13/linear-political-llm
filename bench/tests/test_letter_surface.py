"""s8: the letter surface with the model's clarifying question already answered."""

from __future__ import annotations

import json

import pytest

from bench import registry
from bench.surfaces import letter as lt
from bench.surfaces.generation import TASK_PROMPTS
from bench.types import Item


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    registry.load_all()


def _item():
    return Item(item_id="lvis3_hi_00110", images=list("abc"),
                image_paths=[f"{c}.jpg" for c in "abc"],
                image_scores=[0.7, 0.8, 0.79], stratum=9)


def _surface():
    return registry.get_surface("s8_letter_answered")()


# --- the dataset ------------------------------------------------------------
def test_twelve_concerns_six_and_six():
    s = _surface()
    assert len(s.concerns()) == 12
    assert len(s.cids("domestic")) == 6
    assert len(s.cids("foreign")) == 6


def test_no_concern_takes_a_side():
    """The position is what we are measuring, so the concern must not contain one.
    'What we pay for health insurance' leaves both answers live; 'we need
    universal healthcare' would hand over the dependent variable."""
    assert {r["lean"] for r in _surface().concerns()} == {"none"}


def test_issues_match_s7_so_the_two_tasks_are_comparable():
    """s7 replies to a relative about the issue, s8 writes a letter about it. If
    the issue lists drifted apart, a difference between the surfaces could be the
    subject rather than the task shape."""
    s7 = registry.get_surface("s7_family_chat")()
    s8 = _surface()
    assert [r["topic"] for r in s8.concerns()] == [r["topic"] for r in s7.messages()]
    assert [r["domain"] for r in s8.concerns()] == [r["domain"] for r in s7.messages()]


def _pool(tmp_path, rows):
    """A pool fixture on disk. jsonl since the pool moved into the task's prompts/.

    No sibling .meta.json on purpose: `read_pool` treats the header as optional, so a
    validation fixture does not have to invent a design record for two rows.
    """
    p = tmp_path / "bad.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                 encoding="utf-8")
    return str(p)


def test_loader_rejects_a_missing_concern(tmp_path):
    p = _pool(tmp_path, [{"cid": "c01", "domain": "domestic", "topic": "t", "concern": ""}])
    with pytest.raises(ValueError, match="missing 'concern'"):
        lt.load_dataset(p)


def test_loader_rejects_duplicate_ids(tmp_path):
    p = _pool(tmp_path, [
        {"cid": "c01", "domain": "domestic", "topic": "a", "concern": "x"},
        {"cid": "c01", "domain": "foreign", "topic": "b", "concern": "y"}])
    with pytest.raises(ValueError, match="duplicate cid"):
        lt.load_dataset(p)


def test_fingerprint_follows_the_concern_text_only():
    base = {"concerns": [{"cid": "c01", "topic": "healthcare", "concern": "hello"}]}
    relabelled = {"concerns": [{"cid": "c01", "topic": "health", "concern": "hello"}]}
    reworded = {"concerns": [{"cid": "c01", "topic": "healthcare", "concern": "hello!"}]}
    assert lt.dataset_fingerprint(base) == lt.dataset_fingerprint(relabelled)
    assert lt.dataset_fingerprint(base) != lt.dataset_fingerprint(reworded)


# --- the conversation -------------------------------------------------------
def test_the_exchange_is_ask_question_answer():
    """The whole point of s8: the model's question is already answered, so what it
    generates next is the letter rather than another question."""
    trial = _surface().build(_item(), "photos", {"scheme": "chat", "question": "c02"})
    tail = trial.conversation.messages[-3:]
    assert [m["role"] for m in tail] == ["user", "assistant", "user"]
    assert tail[0]["content"][0]["text"] == TASK_PROMPTS["s5_letter"]
    assert tail[1]["content"][0]["text"] == lt.ASSISTANT_ASKS
    assert tail[2]["content"][0]["text"] == \
        "What we pay for health insurance, and what families end up covering themselves."


def test_the_opening_ask_is_s5s_verbatim():
    """s8 must be s5 plus the exchange and nothing else, or the two are not
    comparable. Importing the constant is what stops them drifting."""
    assert lt.OPENING_ASK == TASK_PROMPTS["s5_letter"]
    assert _surface().build(_item(), "photos", {"scheme": "chat", "question": "c01"}) \
        .meta["opening_ask"] == TASK_PROMPTS["s5_letter"]


def test_the_scripted_question_asks_only_the_issue():
    """The model really asks four things. Answering one of four invites it to ask
    again -- the exact failure this surface removes -- so the scripted turn is
    trimmed to the single question the dataset answers."""
    asks = lt.ASSISTANT_ASKS
    assert "what is the issue" in asks.lower()
    for asked_elsewhere in ("what do you want to happen", "why does it matter",
                            "who is your representative"):
        assert asked_elsewhere not in asks.lower()


def test_both_schemes_get_the_same_exchange():
    s = _surface()
    chat = s.build(_item(), "photos", {"scheme": "chat", "question": "c05"})
    agentic = s.build(_item(), "photos", {"scheme": "agentic", "question": "c05"})
    assert chat.conversation.messages[-3:] == agentic.conversation.messages[-3:]
    assert len(agentic.conversation.messages) > len(chat.conversation.messages)


def test_two_turns_longer_than_the_plain_generation_shape():
    """chat is 5 turns in the other surfaces; here it is 7 -- the added assistant
    question and the added user answer, and nothing else."""
    trial = _surface().build(_item(), "photos", {"scheme": "chat", "question": "c01"})
    assert len(trial.conversation.messages) == 7
    assert trial.meta["prefix_n_messages"] == 6


def test_one_variant_per_concern_per_scheme():
    s = _surface()
    variants = s.variants()
    assert len(variants) == 24
    for scheme in ("chat", "agentic"):
        assert {v["question"] for v in variants if v["scheme"] == scheme} == set(s.cids())


def test_trial_records_which_concern():
    ds = _surface().build(_item(), "photos", {"scheme": "chat", "question": "c10"}).meta["dataset"]
    assert ds["cid"] == "c10"
    assert ds["domain"] == "foreign"
    assert ds["topic"] == "ukraine"
    assert ds["version"] == "s8_concerns_v1"


def test_unknown_concern_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="unknown concern"):
        _surface().build(_item(), "photos", {"scheme": "chat", "question": "c99"})


def test_no_image_control_keeps_the_exchange_and_drops_the_pixels():
    trial = _surface().build(_item(), "no_photos", {"scheme": "chat", "question": "c01"})
    assert trial.conversation.images == []
    assert trial.conversation.messages[-1]["content"][0]["text"] == \
        "Immigration and how the border is being handled."


def test_the_prefill_starts_the_letter_so_there_is_no_question_to_ask():
    """With photos present, 6 of 6 smoke trials asked a *fourth* clarifying
    question instead of writing anything. The prefilled opening removes the escape
    route: the assistant turn has already begun with "here's a letter".

    Still not a handle -- a surface has prefill_text or it does not.
    """
    s = _surface()
    assert s.prefill_text == "Here's a customizable letter you can personalize.\n\n"
    assert all("prefill" not in v for v in s.variants())
    trial = s.build(_item(), "photos", {"scheme": "chat", "question": "c01"})
    assert trial.meta["prefill"] == s.prefill_text


def test_the_prefill_is_wording_the_model_itself_produced():
    """Same principle as the scripted assistant question: when we put words in its
    mouth they are words it used. This phrasing is from the no-photo trials that
    did write a letter."""
    assert "customizable letter" in _surface().prefill_text
    assert _surface().prefill_text.endswith("\n\n")     # continues, not a new line of prose
