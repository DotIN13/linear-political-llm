"""The vLLM adaptor's pure parts, and the gate it is supposed to trip.

No server is contacted: the message conversion and the payload are pure
functions, and the gate needs only the classes.
"""

from bench import registry
from bench.adaptors.base import check_capabilities
from bench.adaptors.vllm_server import (VLLMServerAdaptor, build_payload,
                                        to_openai_messages)
from bench.types import Capability, Conversation, Item, Trial

registry.load_all()

_ITEM = Item(item_id="lvis3_lo_00058", images=["a.jpg", "b.jpg", "c.jpg"],
             image_paths=["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"],
             image_scores=[-0.6, -0.5, -0.7], stratum=0)


def _fake_loader(path):
    return f"data:image/jpeg;base64,FAKE({path})"


def _trial(messages, **kw):
    return Trial(surface="s1_speech", item_id="i1", condition="C",
                 conversation=Conversation(messages=messages), **kw)


# --- capability gate --------------------------------------------------------
def test_vllm_declares_no_activations():
    caps = VLLMServerAdaptor.capabilities
    assert Capability.GENERATE in caps
    assert Capability.IMAGES in caps
    assert Capability.ACTIVATIONS not in caps
    assert Capability.SESSION not in caps


def test_generation_surface_on_vllm_is_degraded_not_blocked():
    surface = registry.get_surface("s1_speech")()
    report = check_capabilities(surface, VLLMServerAdaptor)
    assert report.ok, report.render()          # must not block
    assert report.degraded, report.render()    # but must say it is degraded
    assert any("activation" in d.lower() for d in report.degradations), report.render()


def test_local_hf_still_runs_at_full_fidelity():
    from bench import registry
    from bench.adaptors.local_hf import LocalHFAdaptor
    surface = registry.get_surface("s1_speech")()
    report = check_capabilities(surface, LocalHFAdaptor)
    assert report.ok and not report.degraded, report.render()


# --- message conversion -----------------------------------------------------
def test_chat_turns_map_straight_onto_the_openai_shape():
    msgs = [
        {"role": "user", "content": [{"type": "image", "image": "/a.jpg"},
                                     {"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    out = to_openai_messages(msgs, image_loader=_fake_loader)
    assert out[0]["role"] == "user"
    assert out[0]["content"][0] == {"type": "image_url",
                                    "image_url": {"url": "data:image/jpeg;base64,FAKE(/a.jpg)"}}
    assert out[0]["content"][1] == {"type": "text", "text": "hi"}
    # a pure-text turn collapses to a plain string
    assert out[1] == {"role": "assistant", "content": "hello"}


def test_tool_turns_fold_into_user_turns_keeping_the_image():
    msgs = [
        {"role": "assistant", "content": [{"type": "text", "text": ""}],
         "tool_calls": [{"type": "function",
                         "function": {"name": "view_image",
                                      "arguments": {"path": "/memory/user/img_0417.jpg"}}}]},
        {"role": "tool", "content": [{"type": "image", "image": "/a.jpg"},
                                     {"type": "text", "text": "img_0417.jpg"}]},
    ]
    out = to_openai_messages(msgs, image_loader=_fake_loader)
    # the tool call becomes the literal text Qwen's own template emits
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == (
        '<tool_call>{"name":"view_image","arguments":{"path":"/memory/user/img_0417.jpg"}}</tool_call>')
    # the tool result becomes a user turn, image intact, wrapped in the markers
    assert out[1]["role"] == "user"
    assert out[1]["content"][0]["text"].startswith("<tool_response>")
    assert out[1]["content"][1]["type"] == "image_url"
    assert out[1]["content"][-1]["text"].endswith("</tool_response>")
    # no message keeps role "tool": the OpenAI schema cannot carry an image there
    assert all(m["role"] != "tool" for m in out)


# --- payload ----------------------------------------------------------------
def test_payload_is_greedy_and_carries_max_tokens():
    t = _trial([{"role": "user", "content": [{"type": "text", "text": "q"}]}],
               max_new_tokens=400)
    p = build_payload(t, "m", seed=42, image_loader=_fake_loader)
    assert p["temperature"] == 0.0
    assert p["max_tokens"] == 400
    assert p["seed"] == 42
    assert "continue_final_message" not in p


def test_prefill_becomes_a_continued_assistant_turn():
    """The trial comes from the *real* surface, not a hand-written meta dict.

    The first version of this test wrote ``meta={"prefill_text": ...}`` to match
    what the adaptor read, and the adaptor read a key no surface ever writes --
    so the test passed against a shape that never occurs and the smoke run sent
    no prefill at all. Building the trial through ``surface.build`` is what makes
    the key a contract instead of a coincidence.
    """
    surface = registry.get_surface("s1_speech")()
    t = surface.build(_ITEM, "C", {"scheme": "chat", "prefill": "on"}, seed=42)
    assert t.meta["prefill"] == surface.prefill_text          # the contract
    p = build_payload(t, "m", seed=42, image_loader=_fake_loader)
    assert p["messages"][-1] == {"role": "assistant", "content": surface.prefill_text}
    assert p["continue_final_message"] is True
    assert p["add_generation_prompt"] is False


def test_prefill_off_sends_no_continuation():
    surface = registry.get_surface("s1_speech")()
    t = surface.build(_ITEM, "C", {"scheme": "chat", "prefill": "off"}, seed=42)
    assert t.meta["prefill"] is None
    p = build_payload(t, "m", seed=42, image_loader=_fake_loader)
    assert p["messages"][-1]["role"] == "user"
    assert "continue_final_message" not in p
    assert "add_generation_prompt" not in p


def test_logprobs_are_off_unless_asked():
    t = _trial([{"role": "user", "content": [{"type": "text", "text": "q"}]}], max_new_tokens=8)
    assert "logprobs" not in build_payload(t, "m", seed=1, image_loader=_fake_loader)
    p = build_payload(t, "m", seed=1, logprobs=5, image_loader=_fake_loader)
    assert p["logprobs"] is True and p["top_logprobs"] == 5
