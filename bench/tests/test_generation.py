"""Generation surfaces: six open-ended tasks, schemes as variants, s_pre invariant.

The invariant that matters most (task 3.3): within one scheme the shared prefix
is byte-identical across all six surfaces, so ``s_pre`` -- read at the end of
that prefix -- is identical across surfaces *by construction*.
"""

import pytest

from bench import registry
from bench.adaptors.base import check_capabilities
from bench.store import trial_key
from bench.surfaces.generation import (
    SURFACE_IDS, detect_refusal, extract_mentions_politics, extract_picks,
    extract_topic, load_s3_headlines, shuffled_order, token_set_similarity,
    word_count,
)
from bench.types import Item, Response, sha256_of

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
OTHER = Item(item_id="lvis3_09999", images=["x.jpg"], image_paths=["/tmp/x.jpg"],
             image_scores=[-0.4], stratum=0)

REV = "a1b2c3d4e5f6"


def _prefix_sha(surface, item, scheme, condition="C"):
    trial = surface.build(item, condition, {"scheme": scheme})
    return sha256_of(trial.conversation.messages[:-1])


def _question_sha(surface, item, scheme, condition="C"):
    trial = surface.build(item, condition, {"scheme": scheme})
    return sha256_of(trial.conversation.messages[-1])


def test_six_generation_surfaces_registered():
    assert set(SURFACE_IDS) <= set(registry.surface_names())


@pytest.mark.parametrize("sid", SURFACE_IDS)
def test_generation_surface_shape(sid):
    surface = registry.get_surface(sid)()
    assert surface.family == "generation"
    # round-9: activations moved to `prefers` so the vLLM path is DEGRADED rather
    # than BLOCKED -- the IV (image_mean) is precomputed, so behavioural outcomes
    # survive without the probe. See GenerationSurface's docstring.
    assert {str(c) for c in surface.requires} == {"generate", "images"}
    assert {str(c) for c in surface.prefers} == {"activations", "logprob"}
    # Every surface's variant space is now the conversation style crossed with
    # its own questions -- and nothing else. `question` sits in the dict because
    # trial_key is built from it, not because it is a factor; `prefill` is gone
    # as a dimension entirely (a surface has prefill_text or it does not).
    variants = surface.variants()
    qids = surface.question_ids()
    assert len(variants) == 2 * len(qids)
    assert all(set(v) == {"scheme", "question"} for v in variants), variants
    assert {v["scheme"] for v in variants} == {"chat", "agentic"}
    assert {v["question"] for v in variants} == set(qids)
    # Every generation surface asks one question, except the two that carry a
    # dataset. s1's two-wording contrast is retired: v1 is its only prompt now.
    expected = {"s7_family_chat": 12, "s8_letter_answered": 12}.get(sid, 1)
    assert len(qids) == expected, (sid, qids)
    if sid == "s1_speech":
        assert surface.prefill_text                  # s1 always prefills
        assert "running for Congress" in surface.question()   # the former v1 wording
    else:
        assert surface.prefill_text is None
    assert surface.max_new_tokens > 0
    assert [p.name for p in surface.probe_points(None)] == ["s_pre", "s_gen", "s_img"]
    assert surface.conditions == ["C", "E"]


def test_no_silent_degradation_on_a_probe_less_backend():
    """Acceptance #5, restated for round-9.

    The criterion was never "block"; it was **never drop a column silently**.
    Activations is now a preference, so a probe-less backend runs -- but the gate
    has to say DEGRADED *and name the columns that go missing*, so nobody reads a
    vLLM record as if it had s_gen in it.
    """
    for sid in SURFACE_IDS:
        for adaptor in ("opencode", "vllm"):
            report = check_capabilities(registry.get_surface(sid)(),
                                        registry.get_adaptor(adaptor))
            assert report.ok is True, f"{sid} x {adaptor}: {report.render()}"
            assert report.status == "DEGRADED", f"{sid} x {adaptor}: {report.render()}"
            assert not report.missing_required, report.render()
            blob = " ".join(report.degradations).lower()
            for column in ("s_pre", "s_gen", "s_img"):
                assert column in blob, f"{sid} x {adaptor} must name {column}: {report.render()}"


def test_local_hf_is_the_only_full_fidelity_backend():
    for sid in SURFACE_IDS:
        report = check_capabilities(registry.get_surface(sid)(),
                                    registry.get_adaptor("local_hf"))
        assert report.status == "OK", f"{sid}: {report.render()}"


def test_local_hf_satisfies_the_surface():
    for sid in SURFACE_IDS:
        report = check_capabilities(registry.get_surface(sid)(), registry.get_adaptor("local_hf"))
        assert report.ok is True
        assert report.status == "OK"
        assert report.missing_required == []


@pytest.mark.parametrize("scheme", ["chat", "agentic"])
def test_s_pre_invariant_same_prefix_across_surfaces(scheme):
    """The prefix is identical across all six surfaces, so s_pre is too (task 3.3)."""
    prefixes = {sid: _prefix_sha(registry.get_surface(sid)(), ITEM, scheme) for sid in SURFACE_IDS}
    assert len(set(prefixes.values())) == 1, "the shared prefix must not depend on the surface"

    # ... while the question differs per surface, so only the prefix is shared.
    questions = {sid: _question_sha(registry.get_surface(sid)(), ITEM, scheme) for sid in SURFACE_IDS}
    assert len(set(questions.values())) == 6


@pytest.mark.parametrize("scheme", ["chat", "agentic"])
def test_no_political_word_in_prompt_or_framing(scheme):
    """board-tasks: the prompt and framing must carry no political word -- the
    politics must come out in the answer. (s3's headline *data* is exempt: it is
    the stimulus the model selects from, and its slant is the measurement.)"""
    banned = ["democrat", "republican", "biden", "trump", "liberal", "conservative",
              "gun", "abortion", "immigration", "border", "party", "election",
              "vote", "politic", "policy", "values", "beliefs", "left-wing", "right-wing"]
    for sid in SURFACE_IDS:
        surface = registry.get_surface(sid)()
        assert not any(w in surface.prompt.lower() for w in banned), f"{sid} prompt leaked a political word"
        trial = surface.build(ITEM, "C", {"scheme": scheme})
        prefix_text = " ".join(p["text"] for m in trial.conversation.messages[:-1]
                               for p in m["content"] if p.get("type") == "text").lower()
        for word in banned:
            assert word not in prefix_text, f"{word!r} leaked into the {sid} framing"


def test_scheme_changes_the_trial_key():
    """Acceptance #3a: changing only the scheme must change the key."""
    for sid in SURFACE_IDS:
        k_chat = trial_key(sid, ITEM.item_id, "C", {"scheme": "chat"}, "local_hf", "m", 42, REV)
        k_agent = trial_key(sid, ITEM.item_id, "C", {"scheme": "agentic"}, "local_hf", "m", 42, REV)
        assert k_chat != k_agent


def test_s3_headline_order_changes_the_trial_key():
    """Acceptance #3b: changing only s3's headline order must change the key."""
    order_a = list(range(12))
    order_b = list(reversed(range(12)))
    k_a = trial_key("s3_digest", ITEM.item_id, "C", {"scheme": "chat", "order": order_a},
                    "local_hf", "m", 42, REV)
    k_b = trial_key("s3_digest", ITEM.item_id, "C", {"scheme": "chat", "order": order_b},
                    "local_hf", "m", 42, REV)
    assert k_a != k_b


def test_s3_order_is_deterministic_per_item_and_differs_across_items():
    surface = registry.get_surface("s3_digest")()
    a = surface.build(ITEM, "C", {"scheme": "chat"}).variant["order"]
    a_again = surface.build(ITEM, "C", {"scheme": "chat"}).variant["order"]
    b = surface.build(OTHER, "C", {"scheme": "chat"}).variant["order"]
    assert a == a_again, "the order must be reproducible for resume/dedup"
    assert a != b, "different items must get different orders"
    # A sample of the 24-story pool, not a permutation of it: 12 shown, one side
    # of each of the twelve topics.
    assert len(a) == 12 and len(set(a)) == 12
    assert set(a) <= set(range(24))


def test_s3_order_depends_on_item_and_seed():
    surface = registry.get_surface("s3_digest")()
    a = surface.build(ITEM, "C", {"scheme": "chat"}, seed=42).variant["order"]
    a_same = surface.build(ITEM, "C", {"scheme": "chat"}, seed=42).variant["order"]
    a_diff = surface.build(ITEM, "C", {"scheme": "chat"}, seed=43).variant["order"]
    b = surface.build(OTHER, "C", {"scheme": "chat"}, seed=42).variant["order"]
    assert a == a_same
    assert a != a_diff, "same item + different seed must give a different order"
    assert a != b


def test_s3_question_lists_twelve_headlines_in_the_variant_order():
    surface = registry.get_surface("s3_digest")()
    order = list(reversed(range(12)))
    question = surface.question(order)
    lines = question.splitlines()
    assert lines[0] == surface.prompt
    body = lines[2:]                       # skip the blank separator line
    for i, idx in enumerate(order, start=1):
        h = surface.headlines[idx]
        assert body[i - 1] == f"{i}. {h['outlet']} — {h['headline']}"


def test_s3_attribution_hidden_drops_the_outlet():
    surface = registry.get_surface("s3_digest")()
    order = list(range(12))
    shown = surface.question(order, "shown")
    hidden = surface.question(order, "hidden")
    for i, idx in enumerate(order, start=1):
        h = surface.headlines[idx]
        assert f"{i}. {h['outlet']} — {h['headline']}" in shown
        assert f"{i}. {h['outlet']} — {h['headline']}" not in hidden
        assert f"{i}. {h['headline']}" in hidden


def test_condition_e_is_item_invariant_except_s3():
    for sid in SURFACE_IDS:
        surface = registry.get_surface(sid)()
        expected = sid != "s3_digest"     # s3 shuffles headlines per item
        assert surface.is_item_invariant("E") is expected
        assert surface.is_item_invariant("C") is False


def test_condition_e_drops_images_but_keeps_the_words():
    surface = registry.get_surface("s1_speech")()
    for scheme in ("chat", "agentic"):
        c = surface.build(ITEM, "C", {"scheme": scheme}).conversation
        e = surface.build(ITEM, "E", {"scheme": scheme}).conversation
        assert c.images != [] and e.images == []
        texts_c = [p["text"] for m in c.messages for p in m["content"] if p.get("type") == "text"]
        texts_e = [p["text"] for m in e.messages for p in m["content"] if p.get("type") == "text"]
        assert texts_c == texts_e, "no-image baseline keeps the identical text"


def test_every_message_content_is_a_list():
    """docs/bench/08: a str content crashes transformers' visual scan."""
    for sid in SURFACE_IDS:
        for scheme in ("chat", "agentic"):
            trial = registry.get_surface(sid)().build(ITEM, "C", {"scheme": scheme})
            for message in trial.conversation.messages:
                assert isinstance(message["content"], list), (sid, scheme, message["role"])


# --- deterministic extractors -------------------------------------------------
def test_word_count_and_refusal():
    assert word_count("one two three") == 3
    assert detect_refusal("I'm sorry, but I can't help with that.")
    assert detect_refusal("As an AI, I don't have enough information.")
    assert not detect_refusal("Here is my stump speech. I would lead with my story.")


def test_extract_picks_matches_numbered_headlines_and_never_guesses():
    headlines = load_s3_headlines()
    order = shuffled_order(headlines, "lvis3_00001", 42)
    picks = [1, 4, 7, 10, 11]
    text = "\n".join(
        f"{p}. {headlines[order[p - 1]]['outlet']} — {headlines[order[p - 1]]['headline']}"
        for p in picks
    )
    result = extract_picks(text, headlines, order)
    assert result["parse_ok"] is True
    assert result["match_method"] == "index"
    assert result["n_picked"] == 5
    assert result["picked_positions"] == sorted(picks)
    assert isinstance(result["slant_c_mean"], float)
    assert 0 <= result["n_right"] <= 5
    # only four picked -> parse_ok False, kept but excluded from the DV stats
    short = extract_picks(text.splitlines()[0], headlines, order)
    assert short["parse_ok"] is False
    assert short["slant_c_mean"] is None


def test_token_set_similarity_is_containment():
    assert token_set_similarity(
        "US forces target 3 Iranian oil tankers in response to IRGC missile attack",
        "US forces target 3 Iranian oil tankers",
    ) == pytest.approx(1.0)


def test_extract_topic_left_vs_right():
    assert extract_topic("We must act on climate change and cut carbon emissions.")["topic_slug"] == "climate"
    assert extract_topic("Lower taxes for small businesses now.")["topic_slug"] == "taxes"
    assert extract_topic("The border needs stronger immigration enforcement.")["topic_slug"] == "border"
    assert extract_topic("hello there friend")["topic_slug"] == "other"


def test_extract_mentions_politics():
    assert extract_mentions_politics("They seem civic-minded and politically engaged.")
    assert extract_mentions_politics("She is a loyal Democrat.")
    assert not extract_mentions_politics("She is outdoorsy and kind.")


def test_extract_returns_an_outcome_with_deterministic_fields():
    surface = registry.get_surface("s6_describe")()
    outcome = surface.extract(Response(text="They are civic-minded and kind."),
                              surface.build(ITEM, "C", {"scheme": "chat"}))
    assert outcome.kind == "generation"
    assert outcome.extra["word_count"] == 5
    assert outcome.extra["refusal"] is False
    assert outcome.extra["mentions_politics"] is True


# --- s3 outlet matching (round 9) -------------------------------------------
def test_normalize_outlet_strips_ad_fontes_suffix():
    from bench.surfaces.generation import normalize_outlet
    assert normalize_outlet("Fox News (website)") == "fox news"
    assert normalize_outlet("NPR (website)") == "npr"
    assert normalize_outlet("  The  Epoch Times ") == "the epoch times"
    assert normalize_outlet("") == ""


def test_outlet_matches_is_whole_name_and_unique():
    from bench.surfaces.generation import outlet_matches
    hs = [{"outlet": "Fox News (website)"}, {"outlet": "Fox Business (website)"},
          {"outlet": "NPR (website)"}, {"outlet": "Slate"}]
    # longest name wins on nesting: "Fox Business" must not also match "Fox News"
    assert outlet_matches("**Fox Business** — markets rallied", hs) == [1]
    assert outlet_matches("NPR reported that", hs) == [2]
    # a word that merely contains the name is not a match
    assert outlet_matches("slated for review", hs) == []
    # two distinct outlets in one segment is ambiguous -> caller declines
    assert len(outlet_matches("unlike NPR, Slate argued", hs)) == 2
    assert outlet_matches("nothing here", hs) == []


def test_extract_picks_recovers_a_paraphrase_via_outlet():
    from bench.surfaces.generation import extract_picks
    hs = [{"hid": f"h{i:02d}", "outlet": o, "headline": f"original headline number {i}",
           "slant_c": 0.0, "side": "left" if i % 2 else "right", "topic": f"t{i}"}
          for i, o in enumerate(["Slate", "Fox News (website)", "NPR (website)",
                                 "Daily Wire", "HuffPost", "CNN (website)"])]
    order = list(range(len(hs)))
    # the model paraphrases every headline but keeps the outlet verbatim
    text = ("1. **Slate** — a totally different summary\n"
            "2. **Fox News** — another rewritten sentence\n"
            "3. **NPR** — a third paraphrase entirely\n"
            "4. **Daily Wire** — a fourth one, reworded\n"
            "5. **HuffPost** — and a fifth, also reworded\n")
    out = extract_picks(text, hs, order)
    assert out["parse_ok"] is True
    assert out["picked_hids"] == ["h00", "h01", "h02", "h03", "h04"]
    assert set(out["pick_methods"]) == {"outlet"}
    assert out["match_method"] == "outlet"


# --- s3 order balancing (round 9) -------------------------------------------
def test_a_pinned_order_wins_over_the_seeded_shuffle():
    """Order balancing needs the caller to be able to pin the order exactly."""
    from bench.types import Item
    s3 = registry.get_surface("s3_digest")()
    item = Item(item_id="lvis3_lo_00058", images=["x.jpg", "y.jpg", "z.jpg"],
                image_paths=["a.jpg", "b.jpg", "c.jpg"],
                image_scores=[-0.5, -0.5, -0.6], stratum=0)

    fwd = s3.build(item, "C", {"scheme": "chat"}, seed=42)
    order = list(fwd.variant["order"])
    assert len(order) == 12 and len(set(order)) == 12

    rev = s3.build(item, "C", {"scheme": "chat", "order": list(reversed(order)),
                               "order_arm": "rev"}, seed=42)
    assert rev.variant["order"] == list(reversed(order))
    # the headline at position 1 in fwd is at position 12 in rev -- that is the
    # whole point of the balance
    assert rev.variant["order"][-1] == order[0]
    # and the two are different trials, so they cannot collide on trial_key
    assert trial_key("s3_digest", item.item_id, "C", fwd.variant, "vllm", "m", 42, "rev") != \
           trial_key("s3_digest", item.item_id, "C", rev.variant, "vllm", "m", 42, "rev")


def test_a_surface_without_headlines_ignores_a_pinned_order():
    from bench.types import Item
    s1 = registry.get_surface("s1_speech")()
    item = Item(item_id="i", images=["x.jpg"], image_paths=["a.jpg"],
                image_scores=[0.1], stratum=5)
    t = s1.build(item, "C", {"scheme": "chat", "order": [3, 2, 1]})
    assert t.variant["order"] == [3, 2, 1]      # carried, but unused by the prompt
    assert "3" not in t.meta["question"]


# --- the two conversation templates ----------------------------------------
def _item3():
    from bench.types import Item
    return Item(item_id="x", images=list("abc"), image_paths=["a.jpg", "b.jpg", "c.jpg"],
                image_scores=[0.7, 0.7, 0.7], stratum=9)


def test_chat_share_line_says_what_the_photos_are_of():
    """It used to say only "photos I took recently" -- true and uninformative. The
    persona is the independent variable, so the framing has to name it: where the
    person lives and what they like."""
    from bench.surfaces.generation import SHARE_LINE
    low = SHARE_LINE.lower()
    assert "where i live" in low
    assert "into" in low or "like" in low


def test_neither_template_contains_a_political_word():
    """The red line. Everything before the task's own question is fixed text, and
    if any of it leaned the model, every result would be measuring our wording."""
    from bench.surfaces.generation import (AGENTIC_ACK, AGENTIC_OPENER, ASSISTANT_TURN_1,
                                           ASSISTANT_TURN_2, CHAT_USER_TURN_2, SHARE_LINE,
                                           SYSTEM_AGENTIC)
    blob = " ".join([SHARE_LINE, ASSISTANT_TURN_1, CHAT_USER_TURN_2, ASSISTANT_TURN_2,
                     SYSTEM_AGENTIC, AGENTIC_OPENER, AGENTIC_ACK]).lower()
    for word in ("politic", "vote", "party", "liberal", "conservative", "left", "right",
                 "democrat", "republican", "policy", "government"):
        assert word not in blob, f"{word!r} appears in the fixed template text"


def test_agentic_uses_two_named_memory_directories():
    from bench.surfaces.generation import FILES_BY_DIR, MEMORY_DIRS, build_scheme_messages
    assert MEMORY_DIRS == ["/memory/hometown", "/memory/preferences"]
    msgs, tools = build_scheme_messages("agentic", ["a.jpg", "b.jpg", "c.jpg"], "Q?")
    listed = [m["tool_calls"][0]["function"]["arguments"]["path"]
              for m in msgs if m.get("tool_calls")
              and m["tool_calls"][0]["function"]["name"] == "list_dir"]
    assert listed == MEMORY_DIRS, listed
    # every viewed file sits under the directory it was listed from
    viewed = [m["tool_calls"][0]["function"]["arguments"]["path"]
              for m in msgs if m.get("tool_calls")
              and m["tool_calls"][0]["function"]["name"] == "view_image"]
    expected = [f"{d}/{f}" for d, names in FILES_BY_DIR for f in names]
    assert viewed == expected, viewed


def test_agentic_turn_count_is_pinned():
    """13, not 11: two directories means two list_dir turns. Pinned because the
    turn-count gap against chat (5) is a live confound with the framing, so it
    must not drift silently."""
    from bench.surfaces.generation import build_scheme_messages
    chat, _ = build_scheme_messages("chat", ["a.jpg", "b.jpg", "c.jpg"], "Q?")
    agentic, _ = build_scheme_messages("agentic", ["a.jpg", "b.jpg", "c.jpg"], "Q?")
    assert len(chat) == 5
    assert len(agentic) == 13


def test_each_image_is_attached_exactly_once_in_order():
    """Splitting the files across directories must not duplicate or reorder the
    pixels -- the image sequence is the independent variable."""
    from bench.surfaces.generation import build_scheme_messages
    for scheme in ("chat", "agentic"):
        msgs, _ = build_scheme_messages(scheme, ["a.jpg", "b.jpg", "c.jpg"], "Q?")
        got = [p["image"] for m in msgs for p in m["content"] if p.get("type") == "image"]
        assert got == ["a.jpg", "b.jpg", "c.jpg"], (scheme, got)


def test_the_no_image_baseline_keeps_the_whole_story():
    """Same turns, same directory names, same filenames -- only the pixels go. That
    is what isolates the images from the framing around them."""
    from bench.surfaces.generation import build_scheme_messages
    full, _ = build_scheme_messages("agentic", ["a.jpg", "b.jpg", "c.jpg"], "Q?")
    bare, _ = build_scheme_messages("agentic", [], "Q?")
    assert len(bare) == len(full)
    assert not [p for m in bare for p in m["content"] if p.get("type") == "image"]
    texts = lambda ms: [p.get("text") for m in ms for p in m["content"] if p.get("text")]
    assert texts(bare) == texts(full)
