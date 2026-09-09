"""SQLite cache for judge outputs, keyed on ``(response_hash, judge_id)``.

board step 6: the judge result is a pure function of the answer text and the
judge spec, so it is cached by those two hashes. Rewording a judge prompt only
changes ``judge_id`` -- it never forces the subject model to be re-run. Two
trials that produced the same answer text share one judge result.

The DB lives under ``judge_cache/`` (already in .gitignore). This is the only
judge-side cache; trials stay append-only in ``runs/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench_v2.types import sha256_of

DEFAULT_CACHE_DIR = "judge_cache"


def response_hash(text: str) -> str:
    return sha256_of({"text": text})


class JudgeCache:
    def __init__(self, path: str) -> None:
        # sqlite3 is imported here, not at module import: the GPU job's python
        # environment has a broken libstdc++ that breaks ``import sqlite3``, and
        # the judge cache is never opened on the GPU node (only on the login node).
        import sqlite3

        self.path = path
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS judges ("
            "  response_hash TEXT NOT NULL,"
            "  judge_id TEXT NOT NULL,"
            "  payload TEXT NOT NULL,"
            "  PRIMARY KEY (response_hash, judge_id)"
            ")"
        )
        self._conn.commit()

    def get(self, response_hash: str, judge_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload FROM judges WHERE response_hash = ? AND judge_id = ?",
            (response_hash, judge_id),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, response_hash: str, judge_id: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO judges (response_hash, judge_id, payload) "
            "VALUES (?, ?, ?)",
            (response_hash, judge_id, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "JudgeCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
