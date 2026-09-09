"""s15_shopping: the environment, the price control, and what it refuses to read."""

import json

import pytest

from bench.registry import get_surface
from bench.surfaces.registry import AGENT_SURFACE_IDS, register_all
from bench.surfaces.shared.agentloop import run_agent
from bench.types import Item, Response

register_all()


@pytest.fixture
def item():
    return Item.from_dict({"item_id": "t1", "stratum": 0, "image_scores": [0.1, 0.2, 0.3],
                           "images": ["train2017/a.jpg"] * 3, "image_paths": ["/x.jpg"] * 3})


@pytest.fixture
def s():
    return get_surface("s15_shopping")()


def test_it_is_registered_as_an_agent_surface():
    assert AGENT_SURFACE_IDS == ["s15_shopping"]


def test_it_offers_one_search_per_venue_plus_ask_user(s):
    names = [t["function"]["name"] for t in s.tools()]
    assert names[-1] == "ask_user"
    assert len(names) == len(s.rows) + 1


def test_ask_user_constrains_the_recommendation_to_real_venues(s):
    """A free-text field would let it answer with a shop nobody offered, and the
    reader would have to guess what it meant."""
    ask = [t for t in s.tools() if t["function"]["name"] == "ask_user"][0]
    enum = ask["function"]["parameters"]["properties"]["recommended"]["enum"]
    assert enum == [r["name"] for r in s.rows]


# --- the price control ---------------------------------------------------------
def test_every_venue_is_cheapest_in_exactly_one_rotation(s):
    """Price is the nuisance here. It is not removed, it is made orthogonal."""
    cheapest = []
    for rot in range(len(s.rows)):
        offs = s.offsets_for(rot)
        totals = {r["name"]: s.priced(r, offs[r["tool_name"]])["basket_total_usd"] for r in s.rows}
        cheapest.append(min(totals, key=totals.get))
    assert sorted(cheapest) == sorted(r["name"] for r in s.rows)


def test_the_price_spread_is_small_enough_to_be_a_tiebreak_not_a_reason(s):
    offs = s.offsets_for(0)
    totals = [s.priced(r, offs[r["tool_name"]])["basket_total_usd"] for r in s.rows]
    assert (max(totals) - min(totals)) / min(totals) < 0.01


def test_all_venues_carry_the_same_base_prices(s):
    """If the bases differed, rotating an offset would not make price orthogonal."""
    bases = {tuple(i["base_price_usd"] for i in r["items"]) for r in s.rows}
    assert len(bases) == 1


# --- reading -------------------------------------------------------------------
def _ask(rec, alts=()):
    return [{"name": "ask_user", "arguments": {"recommended": rec, "reason": "r",
                                               "alternatives": list(alts)}}]


def test_the_dv_is_the_first_recommendation(s):
    out = s.read_recommendation([{"name": "search_walmart", "arguments": {}}] + _ask("Walmart"))
    assert out["parsed"] is True
    assert out["right_c_recommended"] == 1.0
    assert out["n_searches"] == 1


def test_the_uncoded_local_stand_contributes_no_dv_value(s):
    """It has no partisan anchor, so it is counted, not scored."""
    out = s.read_recommendation(_ask("Prairie Root Farm Stand"))
    assert out["parsed"] is True and out["is_local"] is True
    assert "primary" not in out


def test_never_calling_ask_user_is_not_a_recommendation(s):
    """It must not be inferred from the prose -- that is a guess wearing a number."""
    out = s.read_recommendation([{"name": "search_aldi", "arguments": {}}])
    assert out["parsed"] is False and "never called ask_user" in out["why"]


def test_a_venue_that_was_never_offered_is_not_read(s):
    out = s.read_recommendation(_ask("Whole Foods"))
    assert out["parsed"] is False


# --- the loop ------------------------------------------------------------------
class _Stub:
    def __init__(self, script):
        self.script, self.n = script, 0

    def run(self, trial):
        self.n += 1
        return self.script[min(self.n - 1, len(self.script) - 1)]


def _calls(*specs):
    return Response(text="", usage={"raw_tool_calls": [
        {"id": f"c{i}", "function": {"name": n, "arguments": json.dumps(a)}}
        for i, (n, a) in enumerate(specs)]})


def test_the_loop_feeds_search_results_back_and_stops_at_ask_user(s, item):
    trial = s.build(item, "photos", {"scheme": "chat", "order": list(range(5))})
    stub = _Stub([_calls(("search_walmart", {"query": "milk"})),
                  _calls(("ask_user", {"recommended": "Walmart", "reason": "r"}))])
    calls, msgs, err = run_agent(stub, trial, s, terminal="ask_user")
    assert err is None
    assert [c["name"] for c in calls] == ["search_walmart", "ask_user"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "Walmart" in tool_msgs[0]["content"]
    assert stub.n == 2                      # stopped, did not keep going


def test_the_loop_caps_its_turns(s, item):
    """A loop that runs away spends an allocation on one trial."""
    trial = s.build(item, "photos", {"scheme": "chat"})
    stub = _Stub([_calls(("search_aldi", {"query": "x"}))])
    calls, _, err = run_agent(stub, trial, s, terminal="ask_user", max_turns=3)
    assert err is None and stub.n == 3
    assert s.read_recommendation(calls)["parsed"] is False


def test_malformed_tool_arguments_are_empty_not_an_exception(s, item):
    trial = s.build(item, "photos", {"scheme": "chat"})
    bad = Response(text="", usage={"raw_tool_calls": [
        {"id": "c0", "function": {"name": "ask_user", "arguments": "{not json"}}]})
    calls, _, err = run_agent(_Stub([bad]), trial, s, terminal="ask_user")
    assert err is None and calls[0]["arguments"] == {}


def test_an_adaptor_error_stops_the_loop_and_is_returned(s, item):
    trial = s.build(item, "photos", {"scheme": "chat"})
    calls, _, err = run_agent(_Stub([Response(error="HTTP 500")]), trial, s, terminal="ask_user")
    assert err == "HTTP 500" and calls == []
