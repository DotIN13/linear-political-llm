"""The run loop a pilot calls instead of writing its own.

A pilot supplies three things and nothing else: the cells it wants, a ``build``
that turns a cell into a ``Trial``, and a ``read`` that turns a ``Response`` into
an ``Outcome``. This module owns resume, the trial key, the record shape and the
manifest, so all 14 pilots cannot drift apart on those.

Three artifacts come out of a run, so nothing measured is only in memory:

* ``trials.jsonl`` -- one row per trial, carrying ``meta`` (what the trial was),
  ``metrics`` (timing, cost, probe) and ``outcome`` (the dependent variable).
* ``transcripts.jsonl`` -- the conversation that was actually sent, plus the
  response text, so a row can be read without chasing its sha.
* ``conversations/<sha2>/<sha>.json`` -- the content-addressed copy, so two
  trials that share a conversation share one file.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bench_v2.store import RunStore, git_rev, measurement_rev, trial_key
from bench_v2.types import Item, Trial, baseline_item

Cell = tuple[str, dict[str, Any], Item]
Build = Callable[[Item, str, dict[str, Any], int], Trial]
Read = Callable[[Any, Trial], Any]
Agent = Callable[[Any, Trial], Any]


def cell_plan(conditions: Sequence[str], variants: Sequence[dict[str, Any]],
              items: Sequence[Item],
              is_item_invariant: Callable[[str], bool]) -> list[Cell]:
    """(condition, variant, item) for every trial a pilot would run.

    An item-invariant condition runs once, against the baseline item, because its
    conversation has no persona in it and would otherwise be byte-identical for
    every item.
    """
    cells: list[Cell] = []
    for condition in conditions:
        for variant in variants:
            variant = dict(variant)
            if is_item_invariant(condition):
                cells.append((condition, variant, baseline_item()))
            else:
                cells.extend((condition, variant, item) for item in items)
    return cells


def run_cells(*, surface: str, cells: Sequence[Cell], build: Build, read: Read,
              adaptor: Any, out_dir: str | Path, seed: int = 42, note: str = "",
              limit_cells: int = 0, run_agent: Agent | None = None,
              verbose: bool = True) -> int:
    """Run the cells, skipping any whose ``trial_key`` is already on disk."""
    out_dir = Path(out_dir)
    rev = measurement_rev(note=f"adaptor={adaptor.name}" + (f" {note}" if note else ""))
    store = RunStore(str(out_dir), conversations_dir=str(out_dir / "conversations"))
    transcripts_path = out_dir / "transcripts.jsonl"
    planned = cells[:limit_cells] if limit_cells else cells
    if verbose:
        print(f"[run] measurement_rev={rev}  {len(planned)} planned, "
              f"{store.n_done} already done -> {store.trials_path}")

    adaptor.setup()
    started = time.time()
    n = 0
    try:
        with transcripts_path.open("a", encoding="utf-8") as transcripts:
            for condition, variant, item in planned:
                trial = build(item, condition, dict(variant), seed)
                key = trial_key(surface, item.item_id, condition, trial.variant,
                                adaptor.name, str(adaptor.model), seed, rev)
                if store.has(key):
                    continue
                store.put_conversation(trial.conversation)
                resp = run_agent(adaptor, trial) if run_agent else adaptor.run(trial)
                outcome = read(resp, trial)

                record = {
                    "trial_key": key, "measurement_rev": rev, "surface": surface,
                    "condition": condition, "variant": trial.variant,
                    "item_id": item.item_id, "is_baseline": item.item_id == "__baseline__",
                    "adaptor": adaptor.name, "model": str(adaptor.model),
                    "seed": seed, "conversation_sha": trial.conversation.sha,
                    "meta": dict(trial.meta),
                    "metrics": {
                        "timing_ms": resp.timing_ms,
                        "cost_usd": resp.cost_usd,
                        "n_messages": len(trial.conversation.messages),
                        "n_images": len(trial.conversation.images),
                        "probe": resp.probe,
                    },
                    "response": resp.to_dict(),
                    "outcome": outcome.to_dict() if hasattr(outcome, "to_dict") else dict(outcome),
                    "error": resp.error,
                }
                store.append(record)
                transcripts.write(json.dumps({
                    "trial_key": key, "surface": surface, "condition": condition,
                    "variant": trial.variant, "item_id": item.item_id,
                    "conversation_sha": trial.conversation.sha,
                    "prefill": (trial.meta or {}).get("prefill"),
                    "messages": trial.conversation.messages,
                    "images": trial.conversation.images,
                    "response_text": resp.text,
                    "error": resp.error,
                }, ensure_ascii=False, sort_keys=True) + "\n")
                transcripts.flush()
                n += 1
                if verbose and n % 10 == 0:
                    print(f"  [{n}/{len(planned)}]", flush=True)
    finally:
        adaptor.teardown()

    store.write_manifest(
        argv=sys.argv, code_rev=git_rev(), adaptor_config=adaptor.describe(),
        started_at=started, finished_at=time.time(), measurement_rev=rev,
        extra={"surface": surface, "n_planned": len(planned)},
    )
    if verbose:
        print(f"[run] wrote {n} new records -> {store.trials_path}")
        print(f"[run] transcripts -> {transcripts_path}")
    return n
