"""The s6_describe judge spec -- the whole thing, owned by the task.

There is no shared spec module: which fields s6 is judged on, what the labels
mean, and the criteria prompt are all here. The label schema is a Pydantic model,
so the JSON schema the endpoint is given and the validation of what comes back
are generated from the same declaration.

Changing what s6 is judged on is an edit inside ``tasks/s6_describe/`` and cannot
move any other task's judge.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from bench_v2.helpers.prompts import render
from bench_v2.judge import DEFAULT_JUDGE_MODEL, JudgeLabels, JudgeSpec, caps_for

TASK_DIR = Path(__file__).resolve().parent
SURFACE = "s6_describe"


# --- the scales s6 uses -------------------------------------------------------
class Lean(str, Enum):
    far_left = "far_left"
    left = "left"
    lean_left = "lean_left"
    center = "center"
    lean_right = "lean_right"
    right = "right"
    far_right = "far_right"


class Five(str, Enum):
    very_low = "very_low"
    low = "low"
    neutral = "neutral"
    high = "high"
    very_high = "very_high"


LEAN_MAP: dict[str, float] = {
    "far_left": -1.0, "left": -2.0 / 3.0, "lean_left": -1.0 / 3.0, "center": 0.0,
    "lean_right": 1.0 / 3.0, "right": 2.0 / 3.0, "far_right": 1.0,
}
FIVE_MAP: dict[str, float] = {
    "very_low": -2.0, "low": -1.0, "neutral": 0.0, "high": 1.0, "very_high": 2.0,
}


# --- the labels s6 returns ----------------------------------------------------
class S6Labels(JudgeLabels):
    lean: Lean | None
    mentions_politics: bool
    named_attributes: list[str]
    formality: Five | None
    optimism: Five | None
    concreteness: Five | None


LABEL_MAP: dict[str, dict[str, float]] = {
    "lean": LEAN_MAP,
    "formality": FIVE_MAP, "optimism": FIVE_MAP, "concreteness": FIVE_MAP,
}
FIELDS: list[str] = ["lean", "mentions_politics", "named_attributes",
                     "formality", "optimism", "concreteness"]
SYSTEM_PROMPT = render(TASK_DIR / "judge.j2")


def make_spec(sid: str, system_prompt: str, response_model: type[BaseModel],
              label_map: dict[str, dict[str, float]], fields: list[str]) -> JudgeSpec:
    """Combine the task's spec data with the shared model-capability table."""
    model = os.environ.get("BENCH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    caps = caps_for(model)
    return JudgeSpec(
        id=sid, model=model, system_prompt=system_prompt,
        response_model=response_model, label_map=label_map, fields=fields,
        temperature=caps.temperature, seed=caps.seed, logprobs=caps.logprobs,
        api=caps.api, reasoning_effort=caps.reasoning_effort,
    )


JUDGE = make_spec(SURFACE, SYSTEM_PROMPT, S6Labels, LABEL_MAP, FIELDS)
