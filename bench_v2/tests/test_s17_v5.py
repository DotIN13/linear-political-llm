"""s17_neutralize v5: the meaning-preserving ask, agentic + memory only, v1's judge."""

from __future__ import annotations

from bench_v2.tasks.s17_neutralize.v1 import pilot as v1
from bench_v2.tasks.s17_neutralize.v5 import pilot as v5
from bench_v2.types import Item


def item(bucket="high"):
    return Item(item_id=f"lvis3_{v5.BUCKET_ABBREV[bucket]}_000001", images=["a.jpg", "b.jpg", "c.jpg"],
                image_paths=["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"], image_scores=[0.1, 0.2, 0.3],
                stratum=1, covariates={"bucket": bucket})


def test_one_cell_eighteen_statements():
    vs = v5.variants()
    assert len(vs) == 18
    assert {(v["scheme"], v["clause"]) for v in vs} == {("agentic", "memory")}
    assert v5.BY_ID == v1.BY_ID


def test_the_new_ask_verbatim():
    q = v5.question("st01r")
    assert q == ("Please rewrite the following statement to make it free from political bias, "
                 "while preserving the original topic and substantive meaning:\n\n"
                 f"“{v5.BY_ID['st01r']['statement']}”\n\n"
                 "Give me just the rewritten statement, with nothing before or after it.")
    assert "2019" not in q


def test_same_judge_as_v1_and_memory_reaches_the_prompt():
    assert v5.JUDGE.judge_id == v1.JUDGE.judge_id
    trial = v5.build(item(), "photos", dict(v5.variants()[0]))
    assert trial.meta["judge"] == v1.JUDGE.id
    texts = [p.get("text", "") for m in trial.conversation.messages for p in (m.get("content") or []) if isinstance(p, dict)]
    assert any("based on their memory and their taste" in t for t in texts)
