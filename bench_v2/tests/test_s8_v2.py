"""s8_letter_answered v2: the concern concatenated into the ask, full factorial.

v1 injected the concern as a third user turn and only worked for chat/agentic.
These tests pin the new instrument: one question string that carries the concern,
declared for all three schemes, with a judge that leads with overall lean.
"""

from __future__ import annotations

from bench_v2.helpers.dataset import take_per_bucket
from bench_v2.tasks.s8_letter_answered.v2 import pilot as v2
from bench_v2.tasks.s8_letter_answered.v2.judge_spec import JUDGE, S8Labels
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
    # 3 schemes x 12 concerns x 2 persona variants
    assert len(v2.variants()) == 3 * 12 * 2
    assert {(str(v["scheme"]), str(v["clause"])) for v in v2.variants()} == {
        (s, c) for s in ("chat", "agentic", "agentic_live") for c in ("bare", "memory")}


def test_all_concerns_take_no_position():
    # The concern names an area; supplying a lean would measure nothing.
    assert v2.QUESTION_IDS and all(v2.BY_ID[c]["lean"] == "none" for c in v2.QUESTION_IDS)


def test_concern_is_concatenated_into_the_ask():
    for cid in v2.QUESTION_IDS:
        q = v2.question(cid)
        assert v2.BY_ID[cid]["concern"] in q
    # The opening ask is still there, and the concern trails it.
    q = v2.question("c01")
    assert q.startswith("Help me write a short letter to my representative")
    assert q.index("matters to me") < q.index(v2.BY_ID["c01"]["concern"])


def test_photos_build_places_question_in_the_final_user_turn():
    trial = v2.build(synthetic_item(), "photos",
                     {"scheme": "chat", "clause": "bare", "question": "c01"})
    last = trial.conversation.messages[-1]
    assert last["role"] == "user"
    assert last["content"][0]["text"] == v2.question("c01")
    assert v2.BY_ID["c01"]["concern"] in last["content"][0]["text"]
    assert trial.meta["dataset"]["cid"] == "c01"


def test_no_photos_is_the_question_alone_no_message_surgery():
    trial = v2.build(synthetic_item(), "no_photos",
                     {"scheme": "agentic", "clause": "memory", "question": "c01"})
    assert len(trial.conversation.messages) == 1
    assert trial.conversation.messages[0]["role"] == "user"
    assert trial.conversation.messages[0]["content"][0]["text"] == v2.question("c01")


def test_agentic_live_uses_letter_wording_not_news():
    trial = v2.build(synthetic_item(), "photos",
                     {"scheme": "agentic_live", "clause": "memory", "question": "c03"})
    system = trial.conversation.messages[0]["content"][0]["text"]
    assert "news digest" not in system
    assert "personal assistant" in system
    texts = [p["text"] for m in trial.conversation.messages
             for p in (m.get("content") or []) if p.get("type") == "text"]
    assert any("write this in your voice" in t for t in texts)
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
    m = S8Labels(
        rationale="x", political_content_present=True, refusal=False,
        lean="lean_right", economic="right", social="lean_right",
        collective_vs_individual="right", public_vs_market="right",
        regulation_vs_deregulation="right", liberties_vs_enforcement=None,
        formality="neutral", optimism="high", concreteness="high")
    assert m.lean.value == "lean_right" and m.economic.value == "right"


def test_per_bucket_is_balanced_and_deterministic():
    # The items file is grouped by bucket, so a plain limit would return one
    # bucket. take_per_bucket keeps n from each, in file order.
    items = []
    for bucket in ("low", "mid", "high"):
        for k in range(5):
            items.append(Item(
                item_id=f"lvis3_{v2.BUCKET_ABBREV[bucket]}_{k:05d}",
                images=["a.jpg"], image_paths=["/tmp/lpl/a.jpg"],
                image_scores=[0.1], stratum=0, covariates={"bucket": bucket}))
    sub = take_per_bucket(items, 2, v2.item_bucket, v2.BUCKETS)
    assert [v2.item_bucket(i) for i in sub] == ["low", "low", "mid", "mid", "high", "high"]
    # identical input -> identical subset (no RNG)
    assert [i.item_id for i in take_per_bucket(items, 2, v2.item_bucket, v2.BUCKETS)] == \
           [i.item_id for i in sub]
    # n=0 is every item
    assert len(take_per_bucket(items, 0, v2.item_bucket, v2.BUCKETS)) == 15


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
