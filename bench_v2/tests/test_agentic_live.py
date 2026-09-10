"""The realistic agentic scheme: the question comes first, then the search.

``agentic`` searches before the ask (its own shape, unchanged); ``agentic_live``
puts the user question first and the machine-inserted list_dir/view_image calls
after it, ending on a tool result so the generation is the final answer.
"""

from __future__ import annotations

from bench_v2.helpers.system_prompt import build_scheme_messages

IMGS = ["a.jpg", "b.jpg", "c.jpg"]


def test_agentic_live_asks_first_then_searches():
    msgs, tools = build_scheme_messages("agentic_live", IMGS, "THE QUESTION", 3, "bare")
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert msgs[1]["content"][0]["text"] == "THE QUESTION"
    # 2 list_dir + 3 view_image = five call/result pairs, then nothing (the model answers)
    assert roles[2:] == ["assistant", "tool"] * 5
    assert all(m.get("tool_calls") for m in msgs if m["role"] == "assistant")
    assert msgs[-1]["role"] == "tool"
    assert tools is not None


def test_agentic_live_memory_opens_with_an_intent_sentence():
    msgs, tools = build_scheme_messages("agentic_live", IMGS, "THE QUESTION", 3, "memory")
    roles = [m["role"] for m in msgs]
    assert roles[:3] == ["system", "user", "assistant"]
    preamble = msgs[2]
    assert not preamble.get("tool_calls")
    assert "memory" in preamble["content"][0]["text"].lower()
    # the announcement comes before any tool call
    assert roles[3:] == ["assistant", "tool"] * 5
    assert msgs[-1]["role"] == "tool"
    assert tools is not None


def test_agentic_still_asks_last():
    msgs, _ = build_scheme_messages("agentic", IMGS, "THE QUESTION", 3, "memory")
    assert msgs[0]["role"] == "user"          # the memory description, unchanged
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"][0]["text"] == "THE QUESTION"


def test_agentic_live_view_image_results_carry_the_pixels():
    msgs, _ = build_scheme_messages("agentic_live", IMGS, "Q", 3, "bare")
    seen = [p["image"] for m in msgs if m["role"] == "tool"
            for p in m["content"] if p.get("type") == "image"]
    assert seen == IMGS


def test_memory_variant_appends_the_instruction():
    bare, _ = build_scheme_messages("agentic_live", IMGS, "Q", 3, "bare")
    mem, _ = build_scheme_messages("agentic_live", IMGS, "Q", 3, "memory")
    b = bare[0]["content"][0]["text"]
    m = mem[0]["content"][0]["text"]
    assert m.startswith(b) and len(m) > len(b)
