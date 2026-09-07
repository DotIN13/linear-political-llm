"""The judge has to find the answer text in either record shape.

`bench run` nests the reply under `response`; a hand-built pilot record writes
it flat as `text`. The judge knew only the nested shape, so it reported
"no generated answers to judge" on 336 real round-9 records -- which reads
exactly like "there is nothing to do".
"""

from bench.cli import _answer_text


def test_nested_shape():
    assert _answer_text({"response": {"text": "hello"}}) == "hello"


def test_flat_shape():
    assert _answer_text({"text": "hello"}) == "hello"


def test_nested_wins_when_both_are_present():
    assert _answer_text({"response": {"text": "nested"}, "text": "flat"}) == "nested"


def test_nothing_to_judge():
    for row in ({}, {"response": None}, {"response": {}}, {"text": ""},
                {"response": {"text": ""}}, {"text": None}, {"text": 42}):
        assert _answer_text(row) is None, row


# --- which surfaces the judge can actually see -----------------------------
def test_every_surface_that_declares_a_rubric_is_judgeable():
    """The bug this pins: `judge_specs()` is keyed by *rubric* id, which happened
    to equal the surface name for the first six surfaces. s7 and s8 deliberately
    reuse s2's rubric so the two are scored on one scale -- and a name-keyed
    lookup found nothing for either, so `bench judge` reported "no generated
    answers to judge" over 1824 real records.
    """
    from bench import registry
    from bench.cli import _spec_for_surface, judgeable_surfaces
    registry.load_all()
    declared = sorted(n for n in registry.surface_names()
                      if getattr(registry.get_surface(n)(), "judge_spec", None))
    assert declared, "no surface declares a rubric; the test has nothing to check"
    assert set(declared) <= set(judgeable_surfaces()), \
        set(declared) - set(judgeable_surfaces())
    for name in declared:
        assert _spec_for_surface(name) is registry.get_surface(name)().judge_spec


def test_the_two_new_surfaces_resolve_to_the_shared_rubric():
    from bench.cli import _spec_for_surface
    a = _spec_for_surface("s7_family_chat")
    b = _spec_for_surface("s8_letter_answered")
    assert a is not None and b is not None
    assert a.judge_id == b.judge_id            # one scale for both
    assert a.id == "s2_proposal"               # reused, not invented


def test_a_surface_with_no_rubric_is_not_judgeable():
    from bench.cli import _spec_for_surface, judgeable_surfaces
    assert _spec_for_surface("s3_digest") is None      # deterministic, no judge
    assert "s3_digest" not in judgeable_surfaces()


def test_an_unknown_surface_name_does_not_raise():
    """Records from a retired surface must not crash the judge step."""
    from bench.cli import _spec_for_surface
    assert _spec_for_surface("s99_does_not_exist") is None
