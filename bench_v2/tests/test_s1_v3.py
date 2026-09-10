"""s1_speech v3: the full factorial, with the task's own agentic wording.

v2's transcripts exposed the delivery wording leaking s3's news-digest text into a
stump speech. v3 fixes that with SCHEME_STYLE and runs chat/agentic/agentic_live.
"""

from __future__ import annotations

from bench_v2.tasks.s1_speech.v3 import pilot as v3
from bench_v2.types import Item


def synthetic_item(bucket: str = "mid") -> Item:
    return Item(
        item_id=f"lvis3_{v3.BUCKET_ABBREV[bucket]}_000001",
        images=["a.jpg", "b.jpg", "c.jpg"],
        image_paths=["/tmp/lpl/a.jpg", "/tmp/lpl/b.jpg", "/tmp/lpl/c.jpg"],
        image_scores=[0.1, 0.2, 0.3],
        stratum=0,
        covariates={"bucket": bucket},
    )


def test_full_factorial():
    variants = v3.variants()
    assert len(variants) == 3 * 2 == 6
    assert {(str(v["scheme"]), str(v["clause"])) for v in variants} == {
        (s, c) for s in ("chat", "agentic", "agentic_live") for c in ("bare", "memory")}


def test_agentic_live_does_not_say_news_digest():
    trial = v3.build(synthetic_item(), "photos", {"scheme": "agentic_live", "clause": "memory"})
    system = trial.conversation.messages[0]["content"][0]["text"]
    assert "news digest" not in system
    assert "personal assistant" in system
    texts = [p["text"] for m in trial.conversation.messages
             for p in (m.get("content") or []) if p.get("type") == "text"]
    assert any("tailor my answer to your taste" in t for t in texts)
    assert not any("recommend news" in t for t in texts)


def test_agentic_live_question_comes_before_the_search():
    trial = v3.build(synthetic_item(), "photos", {"scheme": "agentic_live", "clause": "bare"})
    roles = [m["role"] for m in trial.conversation.messages]
    assert roles[0] == "system" and roles[1] == "user"
    assert trial.conversation.messages[1]["content"][0]["text"] == v3.ASK


def test_ask_is_memory_neutral():
    assert "memory" not in v3.ASK.lower()
