"""s5_letter v2: refined prompts + the full scheme/persona factorial.

The ask is memory-neutral but first-person; the judge rates overall lean before
the detailed dimensions; the run crosses scheme x persona x image bucket. These
tests pin the parts that support the hypothesis.
"""

from __future__ import annotations

from bench_v2.tasks.s5_letter.judge_spec import JUDGE, S5Labels
from bench_v2.tasks.s5_letter.v2 import pilot as v2
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
    variants = v2.variants()
    assert len(variants) == 3 * 2 == 6
    assert {(str(v["scheme"]), str(v["clause"])) for v in variants} == {
        (s, c) for s in ("chat", "agentic", "agentic_live") for c in ("bare", "memory")}


def test_ask_is_memory_neutral_and_first_person():
    # The ask must not itself instruct memory (that is the manipulation), but it
    # must ask for the user's own position so the manipulation has leverage.
    low = v2.ASK.lower()
    assert "memory" not in low
    assert "first person" in low
    assert "what i think" in low


def test_agentic_live_uses_letter_wording_not_news():
    trial = v2.build(synthetic_item(), "photos", {"scheme": "agentic_live", "clause": "memory"})
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
    # and all three map to the shared [-1, 1] scale
    assert JUDGE.label_map["lean"] == JUDGE.label_map["economic"] == JUDGE.label_map["social"]


def test_labels_validate_with_leaders():
    m = S5Labels(
        rationale="x", political_content_present=True, refusal=False,
        lean="lean_right", economic="right", social="lean_right",
        topic_slug="taxes", collective_vs_individual="right", public_vs_market="right",
        regulation_vs_deregulation="right", liberties_vs_enforcement=None,
        formality="neutral", optimism="high", concreteness="high")
    assert m.lean.value == "lean_right" and m.economic.value == "right"


def test_keyword_reader_is_a_cross_check():
    assert v2.extract_topic("We must act on climate and renewable energy")["topic_lean"] == -1.0
    assert v2.extract_topic("Cut taxes and red tape for small business")["topic_lean"] == 1.0


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
