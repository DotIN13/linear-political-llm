"""s3_digest v5: v4 plus an EasyPortrait photo of the user, in every scheme.

The portrait is chosen to match the item's image bucket and is present in both
persona variants, so it is context rather than a stand-in for the memory gate.
"""

from __future__ import annotations

from bench_v2.tasks.s3_digest.v5 import pilot as v5
from bench_v2.types import Item


def synthetic_item(bucket: str = "mid") -> Item:
    return Item(
        item_id=f"lvis3_{v5.BUCKET_ABBREV[bucket]}_000001",
        images=["a.jpg", "b.jpg", "c.jpg"],
        image_paths=["/tmp/lpl/a.jpg", "/tmp/lpl/b.jpg", "/tmp/lpl/c.jpg"],
        image_scores=[0.1, 0.2, 0.3],
        stratum=0,
        covariates={"bucket": bucket},
    )


def _images(trial):
    return [p["image"] for m in trial.conversation.messages
            for p in (m.get("content") or [])
            if isinstance(p, dict) and p.get("type") == "image"]


def test_portrait_bucket_matches_item_bucket():
    for bucket in ("low", "mid", "high"):
        portrait = v5.portrait_for_item(synthetic_item(bucket), 42)
        assert portrait["bucket"] == bucket


def test_portrait_is_fixed_per_item_and_seed():
    item = synthetic_item("high")
    assert v5.portrait_for_item(item, 42) == v5.portrait_for_item(item, 42)
    assert (v5.portrait_for_item(item, 42)["record_id"]
            != v5.portrait_for_item(item, 7)["record_id"])


def test_every_scheme_carries_the_portrait():
    item = synthetic_item("low")
    portrait = v5.portrait_path(v5.portrait_for_item(item, 42))
    for scheme in v5.SCHEMES:
        trial = v5.build(item, "photos", {"scheme": scheme, "clause": "memory"})
        assert portrait in _images(trial), scheme
        assert trial.meta["portrait_bucket"] == "low"
        assert trial.meta["portrait_record_id"]


def test_agentic_reaches_memory_me():
    trial = v5.build(synthetic_item(), "photos", {"scheme": "agentic", "clause": "bare"})
    paths = [tc["function"]["arguments"].get("path")
             for m in trial.conversation.messages if m.get("tool_calls")
             for tc in m["tool_calls"]]
    assert "/memory/me" in paths and "/memory/me/me.jpg" in paths


def test_chat_says_this_is_a_photo_of_me():
    trial = v5.build(synthetic_item(), "photos", {"scheme": "chat", "clause": "bare"})
    first = trial.conversation.messages[0]["content"]
    assert any(p.get("type") == "text" and "photo of me" in p["text"] for p in first)
