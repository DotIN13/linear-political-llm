"""The judge layer: JudgeSpec, an OpenAI-compatible caller, and a sqlite cache.

Three files, one public surface (docs/bench board-judge "要写的代码"):

* ``specs.py``  -- JudgeSpec (model + prompt + schema + label->value map).
* ``caller.py`` -- OpenAI-compatible invoker (strict schema, JSON-mode fallback).
* ``cache.py``  -- sqlite cache keyed on ``(response_hash, judge_id)``.
"""

from bench.judges.cache import JudgeCache, response_hash
from bench.judges.caller import JudgeCaller, JudgeError
from bench.judges.specs import (
    FIVE_MAP, HEDGING_MAP, LEAN_MAP, JudgeSpec, judge_specs,
)

__all__ = [
    "JudgeSpec", "JudgeCaller", "JudgeError", "JudgeCache", "response_hash",
    "judge_specs", "LEAN_MAP", "FIVE_MAP", "HEDGING_MAP",
]
