"""The run loop and the judge aggregation, without a GPU or an API key."""

from __future__ import annotations

import json
import os

from bench_v2.helpers import run as run_helper
from bench_v2.judge import aggregate_labels
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
        return Response(text="Here's an outline for your stump speech: jobs, health, schools.")


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
    assert os.path.exists(os.path.join(out, "manifest.json"))
    assert os.path.exists(os.path.join(out, "conversations"))


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
