"""s7_family_chat v2: full scheme/persona factorial with its own lean judge.

v1 declared two schemes, no persona factor, and used the s2_proposal rubric (no
overall lean). These tests pin the parts that make v2 a hypothesis run.
"""

from __future__ import annotations

from bench_v2.tasks.s7_family_chat.v2 import pilot as v2
from bench_v2.tasks.s7_family_chat.v2.judge_spec import JUDGE, S7Labels
from bench_v2.types import Item


def synthetic_item(bucket: str = "mid") -> Item:
    return Item(
        item_id=f"lvis3_{v2.BUCKET_ABBREV[bucket]}_000001",
        images=["a.jpg", "b.jpg", "c.jpg"],
        image_paths=["/tmp/lpl/a.jpg", "/tmp/lpl/b.jpg", "/tmp/lpl/c.jpg"],
        image_scores=[0.1, 0.2, 0.3],
        stratum=0,
        covariates={"bucket": bucket},
    )


def test_full_factorial():
    # 3 schemes x 12 messages x 2 persona variants
    assert len(v2.variants()) == 3 * 12 * 2
    assert {(str(v["scheme"]), str(v["clause"])) for v in v2.variants()} == {
        (s, c) for s in ("chat", "agentic", "agentic_live") for c in ("bare", "memory")}


def test_messages_take_no_side():
    # The poster never signals a position; the only political input is the photos.
    assert v2.QUESTION_IDS and all(v2.BY_ID[m]["lean"] == "neutral" for m in v2.QUESTION_IDS)


def test_message_reaches_every_scheme_as_the_question():
    # chat/agentic end on the question (final user turn); agentic_live asks first
    # and then searches, so the question is the opening user turn, not the last
    # message. Either way the rendered message is the question the model answers.
    for scheme in v2.SCHEMES:
        trial = v2.build(synthetic_item(), "photos",
                         {"scheme": scheme, "clause": "memory", "question": "m03"})
        q = v2.question("m03")
        assert any(
            any(p.get("type") == "text" and p.get("text") == q
                for p in (m.get("content") or []))
            for m in trial.conversation.messages), scheme
        if scheme in ("chat", "agentic"):
            assert trial.conversation.messages[-1]["role"] == "user", scheme
    assert v2.BY_ID["m03"]["message"] in v2.question("m03")


def test_agentic_live_uses_family_wording_not_news():
    trial = v2.build(synthetic_item(), "photos",
                     {"scheme": "agentic_live", "clause": "memory", "question": "m01"})
    system = trial.conversation.messages[0]["content"][0]["text"]
    assert "news digest" not in system
    assert "personal assistant" in system
    texts = [p["text"] for m in trial.conversation.messages
             for p in (m.get("content") or []) if p.get("type") == "text"]
    assert any("answer the way you would" in t for t in texts)
    assert not any("recommend news" in t for t in texts)


def test_judge_leads_with_overall_lean_before_detailed_dimensions():
    props = list(JUDGE.schema["properties"])
    assert props.index("lean") < props.index("collective_vs_individual")
    assert props.index("economic") < props.index("regulation_vs_deregulation")
    assert props.index("social") < props.index("liberties_vs_enforcement")
    assert JUDGE.label_map["lean"] == JUDGE.label_map["economic"] == JUDGE.label_map["social"]


def test_judge_does_not_reuse_s2_proposal_cache():
    # New fields -> a new judge_id, so it never folds a stale s2_proposal label.
    assert JUDGE.id != "s2_proposal"


def test_labels_validate_with_leaders():
    m = S7Labels(
        rationale="x", political_content_present=True, refusal=False,
        lean="lean_right", economic="right", social="lean_right",
        collective_vs_individual="right", public_vs_market="right",
        regulation_vs_deregulation="right", liberties_vs_enforcement=None,
        formality="low", optimism="neutral", concreteness="high")
    assert m.lean.value == "lean_right" and m.economic.value == "right"


def test_paired_is_within_item_and_drops_incomplete_pairs():
    rows = [
        {"item_id": "lvis3_hi_a", "variant": {"scheme": "chat", "clause": "memory"},
         "judge": {"labels": {"lean": "right"}}},
        {"item_id": "lvis3_hi_a", "variant": {"scheme": "chat", "clause": "bare"},
         "judge": {"labels": {"lean": "center"}}},
        {"item_id": "lvis3_hi_b", "variant": {"scheme": "chat", "clause": "bare"},
         "judge": {"labels": {"lean": "left"}}},
    ]
    out = v2.paired(rows, {"lvis3_hi_a": "high", "lvis3_hi_b": "high"}, v2.lean_of)
    assert out[("chat", "high")][2] == 1
    assert abs(out[("chat", "high")][0] - v2.LEAN_MAP["right"]) < 1e-9
