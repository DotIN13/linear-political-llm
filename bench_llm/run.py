"""The run loop, the store and the trial key.

A pilot hands over its cells, a ``build`` that turns a cell into a ``Trial`` and a
``read`` that turns a ``Response`` into an ``Outcome``. This module owns the rest:

* ``trials.jsonl`` -- one self-contained row per trial (item data, variant, the
  sent conversation, the response, the outcome), appended and fsynced, so a
  killed job loses at most the trial in flight;
* resume -- a trial whose key is already on disk is skipped;
* ``manifest.json`` -- what ran, on what, when.

The key hashes the task, item, variant, adaptor, model, seed and
``instrument_rev``: the content of this package's helpers and of every task's
prompt files. Pilots are left out on purpose, so editing a summary does not
invalidate a trial, while editing an ask does.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from bench_llm.types import Item, Outcome, Response, Trial, canonical_json

HERE = Path(__file__).resolve().parent
INSTRUMENT_GLOBS = ("*.py", "tasks/**/*.j2", "tasks/**/*.txt", "tasks/**/*.json", "tasks/**/*.jsonl")

Cell = Tuple[Dict[str, Any], Item]
Build = Callable[[Item, Dict[str, Any]], Trial]
Read = Callable[[Response, Trial], Outcome]


def instrument_rev(length: int = 12) -> str:
    paths = set()
    for pattern in INSTRUMENT_GLOBS:
        paths.update(glob.glob(str(HERE / pattern), recursive=True))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(os.path.relpath(path, HERE).encode() + b":")
        digest.update(hashlib.sha256(Path(path).read_bytes()).hexdigest().encode())
    return digest.hexdigest()[:length]


def trial_key(task: str, item_id: str, variant: Dict[str, Any], adaptor: str, model: str,
              seed: int, rev: str) -> str:
    payload = "|".join([task, item_id, canonical_json(variant), adaptor, model, str(seed), rev])
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cells(variants: Sequence[Dict[str, Any]], items: Sequence[Item]) -> List[Cell]:
    return [(dict(v), item) for v in variants for item in items]


def read_trials(dirs: Iterable[str | Path]) -> List[Dict[str, Any]]:
    rows = []
    for d in dirs:
        path = Path(d) / "trials.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no records at {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue              # a torn last line from a killed job
    return rows


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def run_cells(*, task: str, cells: Sequence[Cell], build: Build, read: Read, adaptor: Any,
              out_dir: str | Path, workers: int = 1, limit: int = 0, extra: Dict[str, Any] | None = None,
              verbose: bool = True) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    rev = instrument_rev()
    done_keys = {r.get("trial_key") for r in read_trials([out_dir])} if trials_path.exists() else set()
    planned = list(cells[:limit] if limit else cells)
    if verbose:
        print(f"[run] instrument_rev={rev}  {len(planned)} planned, {len(done_keys)} on disk, "
              f"workers={max(1, workers)} -> {trials_path}", flush=True)

    adaptor.setup()
    started, lock, n = time.time(), threading.Lock(), [0]

    with trials_path.open("a", encoding="utf-8") as out:
        def one(cell: Cell) -> None:
            variant, item = cell
            trial = build(item, variant)
            key = trial_key(task, item.item_id, trial.variant, adaptor.name, adaptor.model,
                            adaptor.seed, rev)
            if key in done_keys:
                return
            resp = adaptor.run(trial)
            outcome = read(resp, trial)
            record = {
                "trial_key": key, "instrument_rev": rev, "task": task, "item_id": item.item_id,
                "variant": trial.variant, "adaptor": adaptor.name, "model": adaptor.model,
                "seed": adaptor.seed, "item": item.data, "meta": trial.meta,
                "messages": trial.conversation.messages, "response": resp.to_dict(),
                "outcome": outcome.to_dict(), "error": resp.error,
            }
            with lock:
                if key in done_keys:
                    return
                out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                out.flush()
                os.fsync(out.fileno())
                done_keys.add(key)
                n[0] += 1
                if verbose and n[0] % 50 == 0:
                    print(f"  [{n[0]}/{len(planned)}]", flush=True)

        try:
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for f in as_completed([pool.submit(one, c) for c in planned]):
                        f.result()
            else:
                for c in planned:
                    one(c)
        finally:
            adaptor.teardown()

    manifest = {"task": task, "command": sys.argv, "code_rev": _git_rev(), "instrument_rev": rev,
                "adaptor": adaptor.describe(), "n_planned": len(planned), "n_new": n[0],
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **(extra or {})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    if verbose:
        print(f"[run] wrote {n[0]} new records -> {trials_path}", flush=True)
    return n[0]
