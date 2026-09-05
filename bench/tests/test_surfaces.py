"""Surfaces: eight of them, hard-coded assistant turns, deterministic conversations."""

import pytest

from bench import registry
from bench.surfaces.base import (
    ASSISTANT_TURN_1, ASSISTANT_TURN_2, CONDITIONS, CONDITION_SPEC, SHARE_LINE, USER_TURN_2,
)
from bench.surfaces.choice import CONTROL, POLITICAL
from bench.types import Item

registry.load_all()

ITEM = Item(
    item_id="lvis3_00001",
    images=["train2017/000000000030.jpg", "train2017/000000000034.jpg", "train2017/000000000036.jpg"],
    image_paths=["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"],
    image_scores=[0.51, 0.47, 0.55],
    decile=9,
    covariates={"n_objects": [5, 6, 7]},
    split="explore",
)

# Any of these appearing in the framing would make the manipulation impure.
POLITICAL_WORDS = [
    "democrat", "republican", "biden", "trump", "liberal", "conservative", "gun",
    "abortion", "immigration", "border", "party", "vote", "politic", "policy",
    "values", "beliefs", "left-wing", "right-wing", "election",
]


def test_eight_surfaces_registered():
    assert set(registry.surface_names()) == set(POLITICAL + CONTROL)
    assert len(registry.surface_names()) == 8


@pytest.mark.parametrize("name", POLITICAL + CONTROL)
def test_surface_shape(name):
    surface = registry.get_surface(name)()
    assert surface.name == name
    assert len(surface.candidates) == 2
    assert surface.conditions == CONDITIONS
    assert [p.name for p in surface.probe_points(None)] == ["s_txt", "s_img"]
    assert surface.family == ("political_choice" if name in POLITICAL else "control_choice")


def test_framing_template_contains_no_political_word():
    framing = " ".join([SHARE_LINE, ASSISTANT_TURN_1, USER_TURN_2, ASSISTANT_TURN_2]).lower()
    for word in POLITICAL_WORDS:
        assert word not in framing, f"political word {word!r} leaked into the framing"


@pytest.mark.parametrize("name", POLITICAL + CONTROL)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_conversation_is_deterministic(name, condition):
    surface = registry.get_surface(name)()
    a = surface.build(ITEM, condition)
    b = registry.get_surface(name)().build(ITEM, condition)
    assert a.conversation.sha == b.conversation.sha
    assert a.conversation.messages == b.conversation.messages


def test_different_conditions_give_different_conversations():
    surface = registry.get_surface("vote2020")()
    shas = {c: surface.build(ITEM, c).conversation.sha for c in CONDITIONS}
    assert len(set(shas.values())) == len(CONDITIONS)


def test_different_surfaces_give_different_conversations():
    shas = {name: registry.get_surface(name)().build(ITEM, "C").conversation.sha
            for name in POLITICAL + CONTROL}
    assert len(set(shas.values())) == 8


@pytest.mark.parametrize("condition,n_images", [("A", 3), ("B", 1), ("C", 3), ("D", 1), ("E", 0)])
def test_condition_image_count(condition, n_images):
    trial = registry.get_surface("vote2020")().build(ITEM, condition)
    assert len(trial.conversation.images) == n_images
    parts = [p for m in trial.conversation.messages
             for p in (m["content"] if isinstance(m["content"], list) else [])]
    assert sum(1 for p in parts if p.get("type") == "image") == n_images


@pytest.mark.parametrize("condition", ["C", "D", "E"])
def test_assistant_turns_are_hard_coded_constants(condition):
    trial = registry.get_surface("vote2020")().build(ITEM, condition)
    assistant = [m for m in trial.conversation.messages if m["role"] == "assistant"]
    assert len(assistant) == 2
    assert assistant[0]["content"][0]["text"] == ASSISTANT_TURN_1
    assert assistant[1]["content"][0]["text"] == ASSISTANT_TURN_2
    # and the conversation must end on the user's question, so the chat template
    # is what adds the assistant prefix we read logits at
    assert trial.conversation.messages[-1]["role"] == "user"


@pytest.mark.parametrize("condition", ["A", "B"])
def test_single_turn_conditions_have_no_assistant_turn(condition):
    trial = registry.get_surface("vote2020")().build(ITEM, condition)
    assert [m["role"] for m in trial.conversation.messages] == ["user"]


def test_condition_e_keeps_the_same_words_as_c():
    surface = registry.get_surface("vote2020")()
    texts = {c: [p["text"] for m in surface.build(ITEM, c).conversation.messages
                 for p in m["content"] if p.get("type") == "text"] for c in ("C", "E")}
    assert texts["C"] == texts["E"]


def test_extract_prefers_logprob():
    from bench.types import Response
    surface = registry.get_surface("vote2020")()
    outcome = surface.extract(Response(logprobs={"Biden": -1.83, "Trump": -2.94}))
    assert outcome.kind == "logprob_diff"
    assert outcome.value == pytest.approx(1.11)


def test_extract_falls_back_to_text_then_to_judge():
    from bench.types import NeedsJudge, Response
    surface = registry.get_surface("vote2020")()
    parsed = surface.extract(Response(text="Trump."))
    assert parsed.kind == "choice_text" and parsed.value == -1.0
    assert isinstance(surface.extract(Response(text="I would rather not say.")), NeedsJudge)
    assert isinstance(surface.extract(Response()), NeedsJudge)


def test_condition_spec_covers_all_conditions():
    assert set(CONDITION_SPEC) == set(CONDITIONS)
