"""The judge layer: Pydantic label models, an OpenAI-compatible caller, a sqlite
cache, and a helper that judges a pilot's run directory.

There is no shared spec module. ``JudgeSpec`` (the shape the caller needs and the
cache key it computes) and ``JudgeLabels`` (the three bookkeeping fields every
judge returns) live here; the *spec data* -- the task's label model, its label map
and its criteria prompt -- lives in the task's ``judge_spec.py``.
"""

from bench_v2.judge.cache import JudgeCache, response_hash
from bench_v2.judge.caller import (
    DEFAULT_JUDGE_MODEL, JudgeCaller, JudgeError, JudgeSpec, ModelCaps, caps_for,
)
from bench_v2.judge.run import aggregate_labels, judge_run
from bench_v2.judge.schema import JudgeLabels, strict_schema

__all__ = [
    "JudgeSpec", "JudgeCaller", "JudgeError", "JudgeCache", "response_hash",
    "ModelCaps", "caps_for", "DEFAULT_JUDGE_MODEL",
    "JudgeLabels", "strict_schema",
    "judge_run", "aggregate_labels",
]
