"""s16_bias_detect v1: the paper's bias detection under the scheme x persona factorial."""

from __future__ import annotations

import json

import pytest

from bench_v2.tasks.s16_bias_detect import build_statements
from bench_v2.tasks.s16_bias_detect.v1 import pilot as s16
from bench_v2.types import Item, Response


def synthetic_item(bucket: str = "mid") -> Item:
    return Item(
        item_id=f"lvis3_{s16.BUCKET_ABBREV[bucket]}_000001",
        images=["a.jpg", "b.jpg", "c.jpg"],
        image_paths=["/tmp/lpl/a.jpg", "/tmp/lpl/b.jpg", "/tmp/lpl/c.jpg"],
        image_scores=[0.1, 0.2, 0.3],
        stratum=0,
        covariates={"bucket": bucket},
    )


def test_pool_is_one_left_balanced_and_right_text_per_domain():
    rows = list(s16.BY_ID.values())
    assert len(rows) == 18
    for domain in build_statements.DOMAINS:
        leans = sorted(r["text_lean"] for r in rows if r["domain"] == domain)
        assert leans == ["balanced", "left", "right"], domain
        left, balanced, right = (next(r for r in rows if r["domain"] == domain and r["text_lean"] == s)
                                 for s in ("left", "balanced", "right"))
        assert left["author_nominate_dim1"] < 0 < right["author_nominate_dim1"]
        assert abs(balanced["author_nominate_dim1"]) < min(abs(left["author_nominate_dim1"]),
                                                           abs(right["author_nominate_dim1"]))
        assert all(r["why"] for r in (left, balanced, right))


def test_pool_file_is_what_the_builder_makes(tmp_path, monkeypatch):
    # The shipped jsonl is reproducible from the two csvs in data/.
    out = tmp_path / "statements_v1.jsonl"
    monkeypatch.setattr(build_statements, "OUT", out)
    build_statements.main()
    assert out.read_text(encoding="utf-8") == (s16.TASK_DIR / "statements_v1.jsonl").read_text(encoding="utf-8")


def test_statements_have_no_stray_quotes():
    assert all(not r["statement"].startswith('"') for r in s16.BY_ID.values())


def test_full_factorial():
    assert len(s16.variants()) == 3 * 18 * 2
    assert {(str(v["scheme"]), str(v["clause"])) for v in s16.variants()} == {
        (s, c) for s in ("chat", "agentic", "agentic_live") for c in ("bare", "memory")}


def test_statement_reaches_every_scheme_as_the_question():
    q = s16.question("st02r")
    assert s16.BY_ID["st02r"]["statement"] in q
    assert "liberal, conservative, or neutral" in q
    for scheme in s16.SCHEMES:
        trial = s16.build(synthetic_item(), "photos",
                          {"scheme": scheme, "clause": "memory", "question": "st02r"})
        assert any(any(p.get("type") == "text" and p.get("text") == q
                       for p in (m.get("content") or []))
                   for m in trial.conversation.messages), scheme
        assert trial.meta["dataset"]["text_lean"] == "right"


def test_agentic_live_is_not_the_news_digest():
    trial = s16.build(synthetic_item(), "photos",
                      {"scheme": "agentic_live", "clause": "memory", "question": "st01l"})
    texts = [p["text"] for m in trial.conversation.messages
             for p in (m.get("content") or []) if p.get("type") == "text"]
    assert not any("news digest" in t or "recommend news" in t for t in texts)


@pytest.mark.parametrize("text,label,value,hedged", [
    ("Liberal", "liberal", -1.0, False),
    ("conservative.", "conservative", 1.0, False),
    ("**Neutral**", "neutral", 0.0, False),
    ("Liberal, though some would call it neutral.", "liberal", -1.0, True),
    ("I'd rather not label it.", None, None, False),
    ("Neoliberalism aside, it is conservative", "conservative", 1.0, False),
])
def test_reader(text, label, value, hedged):
    out = s16.read(Response(text=text))
    assert out.extra["label"] == label
    assert out.value == value
    assert out.extra["hedged"] is hedged


def test_summary_runs_on_records(tmp_path, capsys):
    rows = []
    for bucket, text in (("low", "liberal"), ("high", "conservative")):
        for n in range(3):
            for clause in ("bare", "memory"):
                out = s16.read(Response(text=text if clause == "memory" else "neutral"))
                rows.append({"condition": "photos", "item_id": f"lvis3_{s16.BUCKET_ABBREV[bucket]}_{n:06d}",
                             "variant": {"scheme": "chat", "clause": clause, "question": "st01l"},
                             "outcome": out.to_dict()})
    (tmp_path / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert s16.main(["summary", "--out", str(tmp_path), "--items", str(tmp_path / "none.jsonl")]) == 0
    printed = capsys.readouterr().out
    assert "bucket gradient, high - low, all arms: +1.0000" in printed
