"""bench_llm helpers and the party_look v1 pilot. No GPU, no server."""

from __future__ import annotations

import ast
import csv
import json
import math
import os
from pathlib import Path

import pytest

from bench_llm import prompts, readers, run, sources, stats
from bench_llm.adaptors import payload, to_openai
from bench_llm.tasks.party_look.v1 import pilot
from bench_llm.types import Item, Response

PKG = Path(__file__).resolve().parents[1]
FIELDS = ["record_id", "record_name", "image_path", "num_image_tokens", "image_token_mismatch", "image_mean"]


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def root(tmp_path):
    lvis = [{"record_id": f"lvis_{i:03d}", "record_name": f"{i}.jpg", "image_path": f"img/{i}.jpg",
             "num_image_tokens": 100, "image_token_mismatch": 0, "image_mean": (i - 50) / 50}
            for i in range(100)]
    lvis[0]["image_token_mismatch"] = 1
    write_csv(str(tmp_path / pilot.SOURCES["lvis"]), FIELDS, lvis)
    row = dict(num_image_tokens=50, image_token_mismatch=0)
    write_csv(str(tmp_path / pilot.SOURCES["congress"]), FIELDS, [
        {**row, "record_id": "congress_R1", "record_name": "Rep One", "image_path": "c/1.jpg", "image_mean": 0.4},
        {**row, "record_id": "congress_D1", "record_name": "Dem One", "image_path": "c/2.jpg", "image_mean": 0.1},
        {**row, "record_id": "congress_D1", "record_name": "Dem One", "image_path": "c/2.jpg", "image_mean": 0.1}])
    write_csv(str(tmp_path / pilot.LEGISLATORS_CSV), ["bioguide", "party", "congress"], [
        {"bioguide": "R1", "party": "Democrat", "congress": "116"},
        {"bioguide": "R1", "party": "Republican", "congress": "118"},
        {"bioguide": "D1", "party": "Democrat", "congress": "117"}])
    write_csv(str(tmp_path / pilot.NOMINATE_CSV), ["bioguide_id", "nominate_dim1"],
              [{"bioguide_id": "R1", "nominate_dim1": "0.6"}, {"bioguide_id": "D1", "nominate_dim1": "-0.4"}])
    return str(tmp_path)


def item(**data):
    return Item(item_id="x", image_paths=["/tmp/x.jpg"], data={"source": "lvis", **data})


# --- the package stands alone ------------------------------------------------
def test_no_module_imports_bench_v2():
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.split(".")[0] == "bench_v2" for n in names), path


# --- sources helpers, as the pilot assembles them ----------------------------
def test_stratified_seeded_sample(root):
    items, header = pilot.load_source("lvis", strata=5, per_stratum=4, seed=1, root=root)
    assert header["n_pool"] == 99 and len(items) == 20
    assert [sum(1 for i in items if i.data["stratum"] == s) for s in range(5)] == [4] * 5
    again, _ = pilot.load_source("lvis", strata=5, per_stratum=4, seed=1, root=root)
    assert [i.item_id for i in items] == [i.item_id for i in again]
    lo = [i.data["image_mean"] for i in items if i.data["stratum"] == 0]
    hi = [i.data["image_mean"] for i in items if i.data["stratum"] == 4]
    assert max(lo) < min(hi)
    assert items[0].image_paths[0].startswith(root)


def test_congress_gets_latest_party_and_nominate(root):
    items, header = pilot.load_source("congress", strata=2, per_stratum=0, seed=1, root=root)
    by = {i.item_id: i.data for i in items}
    assert header["n_pool"] == 2
    assert by["congress_R1"]["party"] == "Republican" and by["congress_R1"]["nominate_dim1"] == 0.6
    assert by["congress_D1"]["party"] == "Democrat"


# --- prompts -----------------------------------------------------------------
def test_both_orientations_render():
    assert len(pilot.variants()) == 2
    d = pilot.question({"format": "scale", "order": "dem_low"})
    r = pilot.question({"format": "scale", "order": "rep_low"})
    assert "1 = strongly associated with Democrats" in d and "7 = strongly associated with Republicans" in d
    assert "1 = strongly associated with Republicans" in r and "Answer with the number only." in r


def test_the_conversation_and_payload():
    t = pilot.build(item(), {"format": "scale", "order": "dem_low"})
    assert [p["type"] for p in t.conversation.messages[0]["content"]] == ["image", "text"]
    body = payload(t, "m", 42, image_loader=lambda p: "data:x")
    assert body["top_logprobs"] == 20 and body["temperature"] == 0.0 and body["max_tokens"] == 8
    assert body["messages"][0]["content"][0]["image_url"]["url"] == "data:x"
    assert to_openai(prompts.conversation("hi", system="s").messages) == \
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]


# --- reading -----------------------------------------------------------------
def lp(d):
    return {k: math.log(v) for k, v in d.items()}


def test_one_side_in_both_orientations():
    t1 = pilot.build(item(), {"format": "scale", "order": "dem_low"})
    t2 = pilot.build(item(), {"format": "scale", "order": "rep_low"})
    o1 = pilot.read(Response(text="7", logprobs=lp({"7": 0.9, " 6": 0.1})), t1)
    o2 = pilot.read(Response(text="7", logprobs=lp({"7": 0.9, " 6": 0.1})), t2)
    assert o1.value == pytest.approx(0.9 + 0.1 * 2 / 3) and o2.value == pytest.approx(-o1.value)
    assert o1.extra["answer_value"] == 1.0 and o2.extra["answer_value"] == -1.0
    assert o1.extra["log_odds"] is None
    o3 = pilot.read(Response(text="4", logprobs=lp({"4": 0.6, "2": 0.1, "6": 0.3})), t1)
    assert o3.extra["log_odds"] == pytest.approx(math.log(3))


def test_the_written_answer_and_every_digit_logprob_are_kept():
    t = pilot.build(item(), {"format": "scale", "order": "dem_low"})
    raw = lp({"5": 0.5, " 5": 0.2, "4": 0.2, "The": 0.1})
    o = pilot.read(Response(text="5 - the scene is rural.", logprobs=raw), t)
    assert o.extra["answer"] == "5" and o.extra["answer_value"] == pytest.approx(1 / 3)
    assert o.extra["answer_text"].startswith("5 -") and o.extra["answer_is_top"]
    assert o.extra["logprobs"]["5"] == pytest.approx(math.log(0.7))   # "5" and " 5" summed
    assert o.extra["logprobs"]["1"] is None                           # not among the tokens returned
    assert o.extra["top_logprobs"] == raw


def test_an_unread_reply_keeps_its_text():
    t = pilot.build(item(), {"format": "scale", "order": "rep_low"})
    o = pilot.read(Response(text="I'm sorry, I can't tell.", logprobs=lp({"I": 0.9, "4": 0.05})), t)
    assert o.value is None and o.extra["mass"] < pilot.MIN_MASS
    assert o.extra["answer"] is None and o.extra["refusal"]


# --- run loop ----------------------------------------------------------------
class FakeAdaptor:
    name, model, seed = "fake", "m", 1

    def __init__(self):
        self.calls = 0

    def setup(self): pass
    def teardown(self): pass
    def describe(self): return {"name": self.name}

    def run(self, trial):
        self.calls += 1
        return Response(text="1", logprobs=lp({"1": 1.0}))


def test_run_resumes_and_records_are_self_contained(tmp_path):
    items = [Item(item_id=f"i{k}", image_paths=[], data={"source": "lvis", "image_mean": k}) for k in range(3)]
    cells = run.cells(pilot.variants(), items)
    a = FakeAdaptor()
    assert run.run_cells(task=pilot.TASK, cells=cells, build=pilot.build, read=pilot.read,
                         adaptor=a, out_dir=tmp_path, verbose=False) == 6
    assert run.run_cells(task=pilot.TASK, cells=cells, build=pilot.build, read=pilot.read,
                         adaptor=a, out_dir=tmp_path, verbose=False, workers=3) == 0
    assert a.calls == 6
    rows = run.read_trials([tmp_path])
    assert {r["outcome"]["value"] for r in rows} == {-1.0, 1.0}
    assert all(r["item"]["source"] == "lvis" and r["messages"] for r in rows)
    assert json.loads((tmp_path / "manifest.json").read_text())["n_new"] == 0


def test_summary_pairs_each_image_with_its_probe(capsys):
    rows = [{"item_id": f"i{i}", "variant": {"format": "scale", "order": o},
             "item": {"source": "lvis", "image_mean": p},
             "outcome": {"value": p, "extra": {"top_p": 0.5, "log_odds": p, "answer_value": p}}}
            for i, p in enumerate([-0.5, 0.0, 0.5]) for o in ("dem_low", "rep_low")]
    assert pilot.summarize(rows)["lvis"]["value.spearman_probe"] == pytest.approx(1)


def test_stats():
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert stats.auc([3, 4], [1, 2]) == 1 and stats.auc([1, 2], [1, 2]) == pytest.approx(0.5)
    assert stats.ranks([5, 1, 5]) == [2.5, 1, 2.5]
    assert readers.first_mention("liberal then conservative", {"r": ["conservative"], "d": ["liberal"]}) == "d"
