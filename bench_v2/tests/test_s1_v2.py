"""s1_speech v2 is the hypothesis factorial, not a byte-for-byte port.

The point of v2 (PREFERENCES.md) is that the memory instruction and the delivery
mechanism are crossed in, so the paired ``memory - bare`` per scheme x bucket can
be read. These tests pin the shape: the full scheme set, the persona factor, the
memory-neutral ask, the intent sentence where it belongs, and the bucket rule the
proof is stratified on.
"""

from __future__ import annotations

from bench_v2.tasks.s1_speech.v2 import pilot as v2
from bench_v2.types import Item


def synthetic_item(stratum: int = 0, bucket: str = "mid") -> Item:
    return Item(
        item_id=f"lvis3_{v2.BUCKET_ABBREV[bucket]}_000001",
        images=["a.jpg", "b.jpg", "c.jpg"],
        image_paths=["/tmp/lpl/a.jpg", "/tmp/lpl/b.jpg", "/tmp/lpl/c.jpg"],
        image_scores=[0.1, 0.2, 0.3],
        stratum=stratum,
        covariates={"bucket": bucket},
    )


def test_variants_are_the_full_factorial():
    variants = v2.variants()
    assert len(variants) == len(v2.SCHEMES) * len(v2.PERSONA_VARIANTS) == 6
    seen = {(str(v["scheme"]), str(v["clause"])) for v in variants}
    assert seen == {(s, c) for s in v2.SCHEMES for c in v2.PERSONA_VARIANTS}
    assert set(v2.SCHEMES) == {"chat", "agentic", "agentic_live"}


def test_ask_is_memory_neutral():
    # v1's ask says "Based on your memory", which is itself a memory instruction
    # and would contaminate the bare arm. v2's must not.
    assert "memory" not in v2.ASK.lower()
    assert "stump speech" in v2.ASK


def test_agentic_live_memory_carries_the_intent_sentence():
    item = synthetic_item()
    bare = v2.build(item, "photos", {"scheme": "agentic_live", "clause": "bare"})
    mem = v2.build(item, "photos", {"scheme": "agentic_live", "clause": "memory"})
    b_texts = [p["text"] for m in bare.conversation.messages
               for p in m["content"] if p.get("type") == "text"]
    m_texts = [p["text"] for m in mem.conversation.messages
               for p in m["content"] if p.get("type") == "text"]
    assert not any("recommend" in t.lower() for t in b_texts)
    assert any("recommend" in t.lower() for t in m_texts)


def test_no_photos_has_no_images_and_is_item_invariant():
    item = synthetic_item()
    trial = v2.build(item, "no_photos", {"scheme": "chat", "clause": "memory"})
    assert trial.conversation.images == []
    assert v2.is_item_invariant("no_photos")
    assert not v2.is_item_invariant("photos")


def test_item_bucket_reads_the_covariate_then_the_stratum():
    assert v2.item_bucket(synthetic_item(stratum=0, bucket="high")) == "high"
    for stratum, bucket in v2.BUCKET_BY_STRATUM.items():
        item = Item(item_id="x", images=[], image_paths=[], image_scores=[],
                    stratum=stratum, covariates={})
        assert v2.item_bucket(item) == bucket


def test_lean_of_maps_and_skips_nulls():
    assert v2.lean_of({"judge": {"labels": {"lean": "right"}}}) == v2.LEAN_MAP["right"]
    assert v2.lean_of({"judge": {"labels": {"lean": "center"}}}) == 0.0
    assert v2.lean_of({"judge": {"labels": {"lean": None}}}) is None
    assert v2.lean_of({}) is None


def test_paired_is_within_item_and_drops_incomplete_pairs():
    rows = [
        # item A, chat: memory +0.5, bare 0.0 -> +0.5
        {"item_id": "lvis3_hi_a", "variant": {"scheme": "chat", "clause": "memory"},
         "judge": {"labels": {"lean": "right"}}},
        {"item_id": "lvis3_hi_a", "variant": {"scheme": "chat", "clause": "bare"},
         "judge": {"labels": {"lean": "center"}}},
        # item B, chat: only bare -> dropped
        {"item_id": "lvis3_hi_b", "variant": {"scheme": "chat", "clause": "bare"},
         "judge": {"labels": {"lean": "left"}}},
    ]
    bucket_by_item = {"lvis3_hi_a": "high", "lvis3_hi_b": "high"}
    out = v2.paired(rows, bucket_by_item, v2.lean_of)
    assert out[("chat", "high")][2] == 1          # n == 1, B dropped
    assert abs(out[("chat", "high")][0] - v2.LEAN_MAP["right"]) < 1e-9
