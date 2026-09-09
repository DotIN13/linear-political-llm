"""The five forced-choice surfaces: what they show, and what they read back.

The readers are the risk here, not the prompts. A prompt that is wrong is visible the
first time anyone looks at a transcript; a reader that is wrong produces a number of
the right shape and nobody notices. So most of this file is about what happens to an
answer that is malformed, partial, or in an order the surface did not present.
"""

import pytest

from bench.registry import get_surface
from bench.surfaces.registry import CHOICE_SURFACE_IDS, register_all
from bench.surfaces.shared.picks import parse_picks, parse_ranking, parse_tool_call, rank_weights
from bench.types import Item

register_all()


@pytest.fixture
def item():
    return Item.from_dict({
        "item_id": "t1", "stratum": 0, "image_scores": [0.1, 0.2, 0.3],
        "images": ["train2017/a.jpg", "train2017/b.jpg", "train2017/c.jpg"],
        "image_paths": ["/stale/absolute/path.jpg"] * 3,
    })


def surface(sid):
    return get_surface(sid)()


def pool_of(s, qid=None):
    if hasattr(s, "rows"):
        return s.rows
    if hasattr(s, "by_topic"):
        return s.by_topic[qid or s.question_ids()[0]]
    return s.by_scenario[qid or s.question_ids()[0]]


# --- registration --------------------------------------------------------------
@pytest.mark.parametrize("sid", CHOICE_SURFACE_IDS)
def test_every_choice_surface_registers_and_builds(sid, item):
    s = surface(sid)
    qid = s.question_ids()[0]
    trial = s.build(item, "photos", {"scheme": "chat", "question": qid})
    assert trial.surface == sid
    assert trial.conversation.messages
    assert trial.meta["question"]


@pytest.mark.parametrize("sid", CHOICE_SURFACE_IDS)
def test_no_choice_surface_has_a_judge(sid):
    """All five are read by rule. A judge here would be a silent cost and a lie."""
    assert getattr(surface(sid), "judge_spec", None) is None


# --- the question shows what the order says ------------------------------------
@pytest.mark.parametrize("sid", ["s9_neighborhood", "s12_explain", "s11_health", "s14_outfits"])
def test_presentation_order_changes_the_rendered_list(sid):
    s = surface(sid)
    qid = s.question_ids()[0]
    n = len(pool_of(s, qid))
    forward = s.question(list(range(n)), qid=qid)
    reverse = s.question(list(reversed(range(n))), qid=qid)
    assert forward != reverse
    assert forward.splitlines()[2] != reverse.splitlines()[2]


def test_groceries_puts_its_options_in_the_tools_not_the_prompt(item):
    s = surface("s10_groceries")
    trial = s.build(item, "photos", {"scheme": "chat", "order": list(range(8))})
    # api_tools, not tools: meta["tools"] is the agentic transcript's documentation
    # and is never sent. Putting the order tools there read 0 of 72 on the first run.
    names = [t["function"]["name"] for t in trial.meta["api_tools"]]
    assert len(names) == 8
    assert all(n.startswith("order_from_") for n in names)
    # the platform names must not leak into the prompt -- the manipulation is the
    # tool description, and a name in the ask would be a second, uncontrolled copy
    for n in names:
        assert n not in trial.meta["question"]


# --- readers: the good case ----------------------------------------------------
def test_neighbourhood_dv_is_the_mean_right_c_of_the_picks():
    s = surface("s9_neighborhood")
    out = s._deterministic("1, 3, 5\nbecause they are quiet", None)
    picked = [s.rows[i - 1]["right_c"] for i in (1, 3, 5)]
    assert out["parsed"] is True
    assert out["right_pick_mean"] == pytest.approx(sum(picked) / 3)


def test_education_dv_is_relative_to_the_shown_mean():
    s = surface("s12_explain")
    out = s._deterministic("1, 2, 3", None)
    shown = s.by_topic["inflation"]
    picked = sum(r["code_c"] for r in shown[:3]) / 3
    shown_mean = sum(r["code_c"] for r in shown) / len(shown)
    assert out["code_pick_rel"] == pytest.approx(picked - shown_mean)


def test_medicine_reports_the_position_of_the_clinician_route():
    s = surface("s11_health")
    rows = s.by_scenario["sleep"]
    clinician_at = next(i for i, r in enumerate(rows, start=1) if r["route"] == "clinician")
    out = s._deterministic(",".join(str(i) for i in range(1, len(rows) + 1)), None)
    assert out["rank_of_clinician"] == clinician_at


def test_medicine_rank_score_is_zero_for_a_balanced_axis_reversed():
    """The weights sum to zero, so reversing a ranking flips the score's sign."""
    s = surface("s11_health")
    n = len(s.by_scenario["sleep"])
    fwd = s._deterministic(",".join(str(i) for i in range(1, n + 1)), None)
    rev = s._deterministic(",".join(str(i) for i in range(n, 0, -1)), None)
    assert fwd["inst_rank_score"] == pytest.approx(-rev["inst_rank_score"])


def test_clothing_weights_the_picks_by_their_order():
    """s14 is the one task where the order of the picks is part of the answer."""
    s = surface("s14_outfits")
    a = s._deterministic("1, 8, 4", None)
    b = s._deterministic("8, 1, 4", None)
    assert a["right_pick_w"] != b["right_pick_w"]


def test_groceries_reads_the_first_tool_named():
    s = surface("s10_groceries")
    first, second = s.rows[0]["tool_name"], s.rows[5]["tool_name"]
    out = s._deterministic(f"I'll use {first}. Not {second}.", None)
    assert out["tool_called"] == first
    assert out["right_c_called"] == pytest.approx(float(s.rows[0]["right_c"]))


# --- readers: the failures, which are the point --------------------------------
@pytest.mark.parametrize("sid", CHOICE_SURFACE_IDS)
def test_an_unreadable_answer_yields_no_primary(sid):
    """Kept, and excluded from the DV. Never silently zero."""
    out = surface(sid)._deterministic("I would rather not say.", None)
    assert out["parsed"] is False
    assert "primary" not in out


def test_too_few_picks_is_a_parse_failure_not_a_short_list():
    assert parse_picks("1, 2", 3, 10) is None
    assert parse_picks("1, 2, 3", 3, 10) == [1, 2, 3]


def test_a_partial_ranking_is_not_a_ranking():
    assert parse_ranking("2, 1, 4", 6) is None
    assert parse_ranking("2,1,4,3,6,5", 6) == [2, 1, 4, 3, 6, 5]


def test_out_of_range_numbers_are_dropped_before_the_count_is_checked():
    assert parse_picks("11, 12, 13", 3, 10) is None


def test_repeated_numbers_do_not_pad_a_short_answer():
    assert parse_picks("1, 1, 1", 3, 10) is None


def test_a_tool_name_that_is_a_prefix_cannot_shadow_a_longer_one():
    assert parse_tool_call("call order_from_alden", ["order_from_alden", "order_from_al"]) \
        == "order_from_alden"


def test_rank_weights_sum_to_zero_and_span_plus_minus_one():
    for n in (4, 6, 8):
        w = rank_weights(n)
        assert sum(w) == pytest.approx(0.0)
        assert w[0] == pytest.approx(1.0)
        assert w[-1] == pytest.approx(-1.0)


# --- the reader must use the order the trial presented -------------------------
def test_the_dv_follows_the_presented_order_not_the_file_order(item):
    """The bug this guards: reading pick "1" against the pool's file order when the
    trial showed a rotation. Every DV would be wrong, and plausibly so."""
    s = surface("s9_neighborhood")
    n = len(s.rows)
    reversed_order = list(reversed(range(n)))
    trial = s.build(item, "photos", {"scheme": "chat", "order": reversed_order})
    out = s._deterministic("1, 2, 3", trial)
    assert out["picked_ids"] == [s.rows[i]["nid"] for i in reversed_order[:3]]


def test_groceries_asks_the_server_to_actually_call_something(item):
    """tool_choice=required. With it absent the model wrote prose about ordering
    and named no shop at all -- 0 of 72 readable."""
    trial = surface("s10_groceries").build(item, "photos", {"scheme": "chat"})
    assert trial.meta["tool_choice"] == "required"


def test_no_other_surface_sends_api_tools(item):
    """The guard on the change that would be worst: sending the agentic scheme's
    faked transcript tools as real ones would alter the manipulation everywhere."""
    for sid in CHOICE_SURFACE_IDS:
        if sid == "s10_groceries":
            continue
        t = surface(sid).build(item, "photos", {"scheme": "agentic"})
        assert not t.meta.get("api_tools"), sid


def test_groceries_reads_a_structured_tool_call(item):
    """The bug that cost the first run: a real call is in message.tool_calls and
    never appears in the text."""
    from bench.types import Response
    s = surface("s10_groceries")
    name = s.rows[3]["tool_name"]
    trial = s.build(item, "photos", {"scheme": "chat", "order": list(range(8))})
    resp = Response(text="", usage={"tool_calls": [name]})
    extra = s.extract(resp, trial).extra
    assert extra["parsed"] is True
    assert extra["tool_called"] == name
    assert extra["read_from"] == "tool_calls"
    assert extra["right_c_called"] == pytest.approx(float(s.rows[3]["right_c"]))
