"""The assembly layer: parts shared, shapes replaceable, and the prefix invariant.

Two properties have to survive the split into ``prompt_parts`` + ``schemes``.

**The parts are shared, not copied.** The agentic arms differ only in where the
request sits, so their tool chain and their role text must be the *same objects*,
produced by the same functions. If a scheme file ever writes its own
``list_dir``/``view_image`` walk, the contrast becomes two differences at once.

**The prefix is invariant within a scheme.** ``s_pre`` is taken at the end of the
shared prefix -- everything before the request -- so a scheme's prefix must not
depend on the question. That was guaranteed by construction when one function built
every scheme. Now that a task can replace one, it needs asserting.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bench_v2.helpers import prompt_parts as parts
from bench_v2.helpers.schemes import BUILDERS, build_scheme_messages, builder_for

IMGS = ["/items/a.jpg", "/items/b.jpg", "/items/c.jpg"]
SCHEMES = ("chat", "agentic", "agentic_live")


def _walk(messages):
    return [m for m in messages
            if m.get("tool_calls") or m["role"] == "tool"]


def test_one_package_per_scheme_name():
    """``BUILDERS`` is keyed by scheme, so a task's file and directory names say
    which scheme they replace."""
    from bench_v2.helpers.schemes import components

    assert set(BUILDERS) == set(SCHEMES)
    here = Path(__file__).resolve().parents[1] / "helpers" / "schemes"
    for scheme in SCHEMES:
        assert (here / scheme / "__init__.py").is_file(), scheme
        assert scheme in components.COMPONENTS
        # Every component has a default, so a task that overrides none still renders.
        for name in components.COMPONENTS[scheme]:
            assert (here / scheme / f"{name}.j2").is_file(), f"{scheme}/{name}"
    assert not (here / "all.py").exists()


@pytest.mark.parametrize("variant", ["bare", "memory"])
def test_the_agentic_arms_share_one_tool_walk(variant):
    """Same pieces, same objects. The only difference is where the request sits."""
    _a_msgs, a_tools = build_scheme_messages("agentic", IMGS, "Q", 3, variant)
    _l_msgs, l_tools = build_scheme_messages("agentic_live", IMGS, "Q", 3, variant)
    a_msgs, l_msgs = _a_msgs, _l_msgs
    assert a_tools == l_tools == parts.TOOLS
    assert _walk(a_msgs) == _walk(l_msgs)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_request_is_the_only_thing_the_question_moves(scheme):
    """The ``s_pre`` invariant: a scheme's prefix does not depend on the question."""
    one, _ = build_scheme_messages(scheme, IMGS, "question one", 3, "bare")
    two, _ = build_scheme_messages(scheme, IMGS, "question two", 3, "bare")
    assert len(one) == len(two)
    differing = [i for i, (a, b) in enumerate(zip(one, two)) if a != b]
    assert len(differing) == 1, f"{scheme} changed {len(differing)} messages"
    i = differing[0]
    assert one[i]["role"] == "user"
    assert "question one" in str(one[i]) and "question two" in str(two[i])


def test_the_request_sits_where_the_design_says(scheme_of={"chat": -1, "agentic": -1, "agentic_live": 1}):
    for scheme, index in scheme_of.items():
        messages, _ = build_scheme_messages(scheme, IMGS, "Q", 3, "bare")
        assert "Q" in str(messages[index])


def test_the_agentic_role_rides_in_different_slots():
    """Only ``agentic_live`` uses a real system message. Documented, not accidental.

    It matters for what the arms are compared on: ``agentic`` changes both the slot
    the role occupies and the position of the request, so a difference between the
    two arms is not attributable to turn order alone.
    """
    bare = {s: build_scheme_messages(s, IMGS, "Q", 3, "bare")[0] for s in SCHEMES}
    assert bare["chat"][0]["role"] == "user"
    assert bare["agentic"][0]["role"] == "user"
    assert bare["agentic_live"][0]["role"] == "system"


def test_a_task_can_replace_one_scheme_and_keep_the_rest(tmp_path):
    """The override path, exercised end to end on a throwaway task directory."""
    (tmp_path / "chat.py").write_text(textwrap.dedent('''
        """A deliberately different chat shape: one turn, no acknowledgement."""
        from bench_v2.helpers import prompt_parts as parts


        def build(image_paths, question, n_files=3, variant="bare", portrait=None,
                  portrait_name=parts.ME_FILE, *, prompts=None, style=None):
            return [parts.user_text(question)], None
    '''))
    build = builder_for(tmp_path)
    own, _ = build("chat", IMGS, "Q", 3, "bare")
    assert len(own) == 1                      # the task's shape won
    other, tools = build("agentic", IMGS, "Q", 3, "bare")
    assert len(other) == 13 and tools == parts.TOOLS   # the default is untouched


def test_an_unknown_scheme_is_refused_not_defaulted():
    """A typo must not silently produce the chat arm."""
    with pytest.raises(ValueError, match="unknown scheme"):
        build_scheme_messages("chatty", IMGS, "Q", 3, "bare")
    with pytest.raises(ValueError, match="unknown scheme"):
        builder_for("/tmp")("chatty", IMGS, "Q", 3, "bare")
