"""The portrait (EasyPortrait) context, in every scheme, off by default.

The portrait is a photo of the user themselves and is context in *both* persona
variants, so it never stands in for the memory instruction. Delivery follows the
scheme: chat shows it as "This is a photo of me."; the agentic schemes put it in
``/memory/me`` and reach it with a list_dir/view_image pair.
"""

from __future__ import annotations

from bench_v2.helpers.system_prompt import build_scheme_messages

IMGS = ["a.jpg", "b.jpg", "c.jpg"]
ME = "/tmp/me.jpg"


def _images(msgs):
    return [p["image"] for m in msgs for p in (m.get("content") or [])
            if isinstance(p, dict) and p.get("type") == "image"]


def _tool_paths(msgs):
    return [tc["function"]["arguments"].get("path")
            for m in msgs if m.get("tool_calls")
            for tc in m["tool_calls"]]


def test_chat_portrait_is_in_the_first_user_turn_as_me():
    msgs, _ = build_scheme_messages("chat", IMGS, "Q", 3, "bare", portrait=ME)
    first = msgs[0]["content"]
    assert _images(msgs) == IMGS + [ME]
    texts = [p["text"] for p in first if p.get("type") == "text"]
    assert any("photo of me" in t for t in texts)


def test_chat_without_portrait_is_unchanged():
    with_me, _ = build_scheme_messages("chat", IMGS, "Q", 3, "bare", portrait=ME)
    without, _ = build_scheme_messages("chat", IMGS, "Q", 3, "bare")
    assert _images(without) == IMGS
    assert "photo of me" not in str(without)


def test_agentic_puts_the_portrait_in_memory_me():
    msgs, _ = build_scheme_messages("agentic", IMGS, "Q", 3, "bare", portrait=ME)
    paths = _tool_paths(msgs)
    assert "/memory/me" in paths
    assert "/memory/me/me.jpg" in paths
    assert ME in _images(msgs)
    assert "/memory/me" in msgs[0]["content"][0]["text"]


def test_agentic_live_puts_the_portrait_in_memory_me():
    msgs, _ = build_scheme_messages("agentic_live", IMGS, "Q", 3, "bare", portrait=ME)
    paths = _tool_paths(msgs)
    assert "/memory/me" in paths and "/memory/me/me.jpg" in paths
    assert ME in _images(msgs)


def test_agentic_without_portrait_has_no_me_dir():
    msgs, _ = build_scheme_messages("agentic", IMGS, "Q", 3, "memory")
    assert "/memory/me" not in _tool_paths(msgs)
    assert "/memory/me" not in msgs[0]["content"][0]["text"]


def test_style_overrides_the_agentic_wording():
    style = {"agentic_live_system": "You are a speech coach.", "live_intent": "Let me check."}
    msgs, _ = build_scheme_messages("agentic_live", IMGS, "Q", 3, "memory", style=style)
    assert msgs[0]["content"][0]["text"].startswith("You are a speech coach.")
    assert msgs[2]["content"][0]["text"] == "Let me check."


def test_default_agentic_wording_is_the_news_digest_one():
    msgs, _ = build_scheme_messages("agentic_live", IMGS, "Q", 3, "memory")
    assert "news digest agent" in msgs[0]["content"][0]["text"]
