"""Judge a pilot's run directory, and aggregate the labels.

The subject model is never re-run to change a judge: a verdict is a pure function
of the answer text and the spec, so it is cached by ``(response_hash, judge_id)``.
``judged.jsonl`` sits beside ``trials.jsonl`` and is append-only in the same way.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench_v2.judge.cache import JudgeCache, response_hash
from bench_v2.judge.caller import JudgeCaller, JudgeError, JudgeSpec
from bench_v2.paths import judge_cache_path


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def judge_run(run_dir: str | Path, spec: JudgeSpec,
              cache_path: str | Path | None = None,
              limit: int = 0, verbose: bool = True) -> int:
    """Judge every answer in ``run_dir/trials.jsonl``, appending to ``judged.jsonl``.

    Returns the number of newly written rows. A row already present for this
    ``(trial_key, judge_id)`` is skipped, so re-running after a crash resumes.
    """
    run_dir = Path(run_dir)
    trials = _read_jsonl(run_dir / "trials.jsonl")
    out_path = run_dir / "judged.jsonl"
    seen = {(r.get("trial_key"), r.get("judge_id")) for r in _read_jsonl(out_path)}
    planned = trials[:limit] if limit else trials
    if verbose:
        print(f"[judge] {len(planned)} trials, {len(seen)} already judged -> {out_path}")
    if not planned:
        return 0
    cache = JudgeCache(cache_path or judge_cache_path())
    caller = JudgeCaller(spec)

    n = 0
    with out_path.open("a", encoding="utf-8") as out:
        for record in planned:
            key = record.get("trial_key")
            if (key, spec.judge_id) in seen:
                continue
            text = ((record.get("response") or {}).get("text") or "").strip()
            if not text:
                continue
            digest = response_hash(text)
            payload = cache.get(digest, spec.judge_id)
            if payload is None:
                try:
                    payload = caller.call(text)
                except JudgeError as exc:
                    payload = {"judge_id": spec.judge_id, "model": spec.model,
                               "labels": None, "error": str(exc)}
                else:
                    cache.put(digest, spec.judge_id, payload)
            row = {
                "trial_key": key, "judge_id": spec.judge_id, "model": spec.model,
                "labels": payload.get("labels"), "error": payload.get("error"),
            }
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            n += 1
            if verbose and n % 10 == 0:
                print(f"  [{n}/{len(planned)}]", flush=True)
    cache.close()
    if verbose:
        print(f"[judge] wrote {n} new rows")
    return n


def aggregate_labels(run_dir: str | Path, spec: JudgeSpec) -> dict[str, dict[str, Any]]:
    """Mean of every mapped label field over the judged rows.

    Labels that are null or unparseable are excluded from that field's mean but
    counted in ``n``, so a field with heavy attrition is visible rather than
    silently averaged over fewer rows.
    """
    rows = _read_jsonl(Path(run_dir) / "judged.jsonl")
    values: dict[str, list[float]] = defaultdict(list)
    n_rows = 0
    for row in rows:
        labels = row.get("labels") or {}
        if not labels:
            continue
        n_rows += 1
        for field, mapping in spec.label_map.items():
            label = labels.get(field)
            if label in mapping:
                values[field].append(float(mapping[label]))
    out: dict[str, dict[str, Any]] = {}
    for field in spec.label_map:
        vals = values.get(field, [])
        out[field] = {
            "n": len(vals), "mean": (sum(vals) / len(vals)) if vals else None,
        }
    out["_rows"] = {"n": n_rows, "mean": None}
    return out
