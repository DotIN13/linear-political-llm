"""The run loop and the judge aggregation, without a GPU or an API key."""

from __future__ import annotations

import json
import os
import time

from bench_v2.helpers import run as run_helper
from bench_v2.judge import aggregate_labels, attach_judge
from bench_v2.tasks.s1_speech.v1 import pilot as v1
from bench_v2.types import Response

from bench_v2.tests.test_s1_v1_parity import synthetic_item


class DummyAdaptor:
    name = "dummy"
    model = "dummy-1"
    capabilities = frozenset()

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def describe(self):
        return {"name": self.name, "model": self.model}

    def run(self, trial):
        return Response(
            text="Here's an outline for your stump speech: jobs, health, schools.",
            probe={"s_pre": [1.0, 2.0]}, timing_ms=12.5, cost_usd=0.0,
        )


class SlowAdaptor(DummyAdaptor):
    """Sleeps so several calls really are in flight at once under the pool."""

    def run(self, trial):
        time.sleep(0.05)
        return super().run(trial)


def test_run_cells_writes_and_resumes(tmp_path):
    cells = run_helper.cell_plan(("no_photos",), v1.variants(), [synthetic_item()],
                                 v1.is_item_invariant)
    out = str(tmp_path)

    first = DummyAdaptor()
    n1 = run_helper.run_cells(surface="s1_speech", cells=cells, build=v1.build,
                              read=v1.read, adaptor=first, out_dir=out, seed=42,
                              verbose=False)
    assert n1 == len(cells) == 2  # no_photos is item-invariant: one baseline item

    second = DummyAdaptor()
    n2 = run_helper.run_cells(surface="s1_speech", cells=cells, build=v1.build,
                              read=v1.read, adaptor=second, out_dir=out, seed=42,
                              verbose=False)
    assert n2 == 0, "a resumed run must not re-run already-stored trial keys"

    rows = [json.loads(line) for line in open(os.path.join(out, "trials.jsonl")) if line.strip()]
    assert len(rows) == len(cells)
    assert all(row["outcome"]["extra"]["word_count"] == 10 for row in rows)
    assert all(row["response"]["text"] for row in rows)

    # meta and metrics ride along, so a row says what it was and what it cost.
    for row in rows:
        assert row["meta"]["scheme"] in {"chat", "agentic"}
        assert row["meta"]["judge"] == "s1_speech"
        assert row["metrics"]["timing_ms"] == 12.5
        assert row["metrics"]["n_messages"] > 0
        assert row["metrics"]["probe"] == {"s_pre": [1.0, 2.0]}

    # transcripts.jsonl carries the conversation that was sent, not just its sha.
    transcripts = [json.loads(line) for line in
                   open(os.path.join(out, "transcripts.jsonl")) if line.strip()]
    assert len(transcripts) == len(cells)
    assert all(t["messages"] and t["response_text"] for t in transcripts)
    assert all(t["conversation_sha"] for t in transcripts)

    assert os.path.exists(os.path.join(out, "manifest.json"))
    assert os.path.exists(os.path.join(out, "conversations"))


def test_run_cells_parallel_writes_each_once(tmp_path):
    cells = run_helper.cell_plan(("no_photos",), v1.variants(), [synthetic_item()],
                                 v1.is_item_invariant)
    out = str(tmp_path)
    n = run_helper.run_cells(surface="s1_speech", cells=cells, build=v1.build,
                             read=v1.read, adaptor=SlowAdaptor(), out_dir=out,
                             seed=42, verbose=False, workers=4)
    assert n == len(cells)
    rows = [json.loads(line) for line in
            open(os.path.join(out, "trials.jsonl")) if line.strip()]
    transcripts = [json.loads(line) for line in
                   open(os.path.join(out, "transcripts.jsonl")) if line.strip()]
    keys = [row["trial_key"] for row in rows]
    assert len(keys) == len(set(keys)) == len(cells)  # no dups, no losses under the pool
    assert len(transcripts) == len(cells)
    assert all(row["response"]["text"] for row in rows)


def test_attach_judge_folds_labels_into_trials(tmp_path):
    cells = run_helper.cell_plan(("no_photos",), v1.variants(), [synthetic_item()],
                                 v1.is_item_invariant)
    out = str(tmp_path)
    run_helper.run_cells(surface="s1_speech", cells=cells, build=v1.build,
                         read=v1.read, adaptor=DummyAdaptor(), out_dir=out,
                         seed=42, verbose=False)
    rows = [json.loads(line) for line in
            open(os.path.join(out, "trials.jsonl")) if line.strip()]

    # A judged row for the first trial only; the second must stay untouched.
    with open(os.path.join(out, "judged.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "trial_key": rows[0]["trial_key"], "judge_id": v1.JUDGE.judge_id,
            "labels": {"lean": "left", "refusal": False},
        }) + "\n")

    assert attach_judge(out, v1.JUDGE) == 1
    after = [json.loads(line) for line in
             open(os.path.join(out, "trials.jsonl")) if line.strip()]
    assert after[0]["judge"]["labels"]["lean"] == "left"
    assert after[0]["judge"]["judge_id"] == v1.JUDGE.judge_id
    assert "judge" not in after[1]
    assert [r["trial_key"] for r in after] == [r["trial_key"] for r in rows]


def test_aggregate_labels_maps_and_counts(tmp_path):
    rows = [
        {"trial_key": "a", "judge_id": v1.JUDGE.judge_id,
         "labels": {"lean": "left", "formality": "high", "refusal": False}},
        {"trial_key": "b", "judge_id": v1.JUDGE.judge_id,
         "labels": {"lean": "far_right", "formality": None, "refusal": False}},
        {"trial_key": "c", "judge_id": v1.JUDGE.judge_id, "labels": None},
    ]
    with open(os.path.join(tmp_path, "judged.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    stats = aggregate_labels(str(tmp_path), v1.JUDGE)
    assert stats["_rows"]["n"] == 2
    assert stats["lean"]["n"] == 2
    assert abs(stats["lean"]["mean"] - ((-2 / 3) + 1.0) / 2) < 1e-9
    assert stats["formality"]["n"] == 1  # the null is excluded, not treated as zero
