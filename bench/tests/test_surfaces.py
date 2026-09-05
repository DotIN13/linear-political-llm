"""Surfaces: eight of them, hard-coded assistant turns, deterministic conversations."""

import pytest

from bench import registry
from bench.surfaces.base import (
    ANSWER_INSTRUCTION, ASSISTANT_TURN_1, ASSISTANT_TURN_2, CONDITIONS, CONDITION_SPEC,
    LETTERS, ORDERS, SHARE_LINE, USER_TURN_2, order_to_options, validate_variant_space,
)
from bench.surfaces.choice import CONTROL, POLITICAL
from bench.types import Item

registry.load_all()

ITEM = Item(
    item_id="lvis3_00001",
    images=["train2017/000000000030.jpg", "train2017/000000000034.jpg", "train2017/000000000036.jpg"],
    image_paths=["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"],
    image_scores=[0.51, 0.47, 0.55],
    stratum=9,
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
    assert len(surface.options) == 2
    assert surface.candidates == LETTERS      # the letters are what gets measured
    assert len(surface.phrasings) == 3        # docs/bench/03: three rewordings per question
    assert surface.conditions == CONDITIONS
    assert [p.name for p in surface.probe_points(None)] == ["s_txt", "s_img"]
    assert surface.family == ("political_choice" if name in POLITICAL else "control_choice")


@pytest.mark.parametrize("name", POLITICAL + CONTROL)
def test_question_lists_the_options_and_asks_for_a_letter(name):
    surface = registry.get_surface(name)()
    for phrasing in range(len(surface.phrasings)):
        for order in ORDERS:
            question = surface.question({"phrasing": phrasing, "order": order})
            lines = question.splitlines()
            assert lines[0] == surface.phrasings[phrasing]
            expected = order_to_options(surface.options, order)
            assert lines[1] == f"A. {expected[0]}"
            assert lines[2] == f"B. {expected[1]}"
            assert lines[-1] == ANSWER_INSTRUCTION


@pytest.mark.parametrize("name", POLITICAL + CONTROL)
def test_variant_space_is_declared_by_the_surface_and_canonical(name):
    surface = registry.get_surface(name)()
    assert validate_variant_space(surface) == []
    variants = surface.variants()
    assert {v["order"] for v in variants} == set(ORDERS)   # order balancing is mandatory
    assert all(set(v) == {"phrasing", "order"} for v in variants)


def test_a_mistyped_variant_space_is_caught():
    class Broken(registry.get_surface("vote2020")):
        def variants(self):
            return [{"phrasing": 0, "order": "ab"}, {"phrasing": 0, "ordering": "ba"}]

    problems = validate_variant_space(Broken())
    assert problems, "a typo'd variant key must not pass silently"
    assert any("unknown keys" in p for p in problems)


def test_phrasing_index_out_of_range_raises():
    surface = registry.get_surface("vote2020")()
    with pytest.raises(ValueError):
        surface.question({"phrasing": 99, "order": "ab"})


def test_framing_template_contains_no_political_word():
    framing = " ".join([SHARE_LINE, ASSISTANT_TURN_1, USER_TURN_2, ASSISTANT_TURN_2]).lower()
    for word in POLITICAL_WORDS:
        assert word not in framing, f"political word {word!r} leaked into the framing"


@pytest.mark.parametrize("name", POLITICAL + CONTROL)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_conversation_is_deterministic(name, condition):
    surface = registry.get_surface(name)()
    a = surface.build(ITEM, condition, {"phrasing": 0, "order": "ab"})
    b = registry.get_surface(name)().build(ITEM, condition, {"phrasing": 0, "order": "ab"})
    assert a.conversation.sha == b.conversation.sha
    assert a.conversation.messages == b.conversation.messages


def test_different_conditions_give_different_conversations():
    surface = registry.get_surface("vote2020")()
    shas = {c: surface.build(ITEM, c, {"phrasing": 0, "order": "ab"}).conversation.sha
            for c in CONDITIONS}
    assert len(set(shas.values())) == len(CONDITIONS)


def test_different_surfaces_give_different_conversations():
    shas = {name: registry.get_surface(name)().build(
        ITEM, "C", {"phrasing": 0, "order": "ab"}).conversation.sha
        for name in POLITICAL + CONTROL}
    assert len(set(shas.values())) == 8


def test_every_variant_gives_a_different_conversation():
    """Phrasing and order both change the text, so they must change the sha too."""
    surface = registry.get_surface("vote2020")()
    variants = [{"phrasing": p, "order": o} for p in range(3) for o in ORDERS]
    shas = {surface.build(ITEM, "C", v).conversation.sha for v in variants}
    assert len(shas) == len(variants)


@pytest.mark.parametrize("condition,n_images", [("A", 3), ("B", 1), ("C", 3), ("D", 1), ("E", 0)])
def test_condition_image_count(condition, n_images):
    trial = registry.get_surface("vote2020")().build(ITEM, condition, {"phrasing": 0, "order": "ab"})
    assert len(trial.conversation.images) == n_images
    parts = [p for m in trial.conversation.messages
             for p in (m["content"] if isinstance(m["content"], list) else [])]
    assert sum(1 for p in parts if p.get("type") == "image") == n_images


@pytest.mark.parametrize("condition", ["C", "D", "E"])
def test_assistant_turns_are_hard_coded_constants(condition):
    trial = registry.get_surface("vote2020")().build(ITEM, condition, {"phrasing": 0, "order": "ab"})
    assistant = [m for m in trial.conversation.messages if m["role"] == "assistant"]
    assert len(assistant) == 2
    assert assistant[0]["content"][0]["text"] == ASSISTANT_TURN_1
    assert assistant[1]["content"][0]["text"] == ASSISTANT_TURN_2
    # and the conversation must end on the user's question, so the chat template
    # is what adds the assistant prefix we read logits at
    assert trial.conversation.messages[-1]["role"] == "user"


@pytest.mark.parametrize("condition", ["A", "B"])
def test_single_turn_conditions_have_no_assistant_turn(condition):
    trial = registry.get_surface("vote2020")().build(ITEM, condition, {"phrasing": 0, "order": "ab"})
    assert [m["role"] for m in trial.conversation.messages] == ["user"]


def test_condition_e_keeps_the_same_words_as_c():
    surface = registry.get_surface("vote2020")()
    texts = {c: [p["text"] for m in surface.build(ITEM, c, {"phrasing": 0, "order": "ab"}).conversation.messages
                 for p in m["content"] if p.get("type") == "text"] for c in ("C", "E")}
    assert texts["C"] == texts["E"]


def test_extract_prefers_logprob_and_reorients_the_ba_order():
    """A "ba" reading is negated before it is averaged with the "ab" reading."""
    from bench.types import Response
    surface = registry.get_surface("vote2020")()
    response = Response(logprobs={"A": -1.83, "B": -2.94})

    ab = surface.extract(response, surface.build(ITEM, "C", {"phrasing": 0, "order": "ab"}))
    assert ab.kind == "logprob_diff"
    assert ab.value == pytest.approx(1.11)          # A=Biden here, so + means Biden
    assert ab.extra["letter_to_option"] == {"A": "Biden", "B": "Trump"}

    ba = surface.extract(response, surface.build(ITEM, "C", {"phrasing": 0, "order": "ba"}))
    assert ba.value == pytest.approx(-1.11)         # same letters, A=Trump, so + means Trump
    assert ba.extra["raw_letter_diff"] == pytest.approx(1.11)
    assert ba.extra["letter_to_option"] == {"A": "Trump", "B": "Biden"}
    # both are oriented onto options[0], so their difference is the position bias
    assert ab.value - ba.value == pytest.approx(2.22)


def test_position_bias_is_zero_when_the_model_ignores_position():
    """If the model answers by content only, the two orders mirror each other."""
    from bench.types import Response
    surface = registry.get_surface("vote2020")()
    ab = surface.extract(Response(logprobs={"A": -1.0, "B": -2.0}),
                         surface.build(ITEM, "C", {"phrasing": 0, "order": "ab"}))
    ba = surface.extract(Response(logprobs={"A": -2.0, "B": -1.0}),
                         surface.build(ITEM, "C", {"phrasing": 0, "order": "ba"}))
    assert ab.value == pytest.approx(1.0) and ba.value == pytest.approx(1.0)
    assert ab.value - ba.value == pytest.approx(0.0)


def test_extract_falls_back_to_text_then_to_judge():
    from bench.types import NeedsJudge, Response
    surface = registry.get_surface("vote2020")()
    trial_ab = surface.build(ITEM, "C", {"phrasing": 0, "order": "ab"})
    trial_ba = surface.build(ITEM, "C", {"phrasing": 0, "order": "ba"})
    assert surface.extract(Response(text="B"), trial_ab).value == -1.0   # B = Trump
    assert surface.extract(Response(text="B"), trial_ba).value == +1.0   # B = Biden
    assert surface.extract(Response(text="Trump."), trial_ab).value == -1.0
    assert isinstance(surface.extract(Response(text="I would rather not say."), trial_ab),
                      NeedsJudge)
    assert isinstance(surface.extract(Response(), trial_ab), NeedsJudge)


@pytest.mark.parametrize("name", POLITICAL + CONTROL)
def test_condition_e_is_item_invariant_and_the_others_are_not(name):
    """Task C: E shows no image, so every item would give a byte-identical trial."""
    surface = registry.get_surface(name)()
    assert surface.is_item_invariant("E") is True
    for condition in ("A", "B", "C", "D"):
        assert surface.is_item_invariant(condition) is False

    other = Item(item_id="lvis3_09999", images=["x.jpg"], image_paths=["/tmp/x.jpg"],
                 image_scores=[-0.4], stratum=0)
    variant = {"phrasing": 0, "order": "ab"}
    assert (surface.build(ITEM, "E", variant).conversation.sha
            == surface.build(other, "E", variant).conversation.sha)
    assert (surface.build(ITEM, "C", variant).conversation.sha
            != surface.build(other, "C", variant).conversation.sha)


def test_condition_spec_covers_all_conditions():
    assert set(CONDITION_SPEC) == set(CONDITIONS)
