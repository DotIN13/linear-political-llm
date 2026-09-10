"""The s5_letter judge spec -- the whole thing, owned by the task.

There is no shared spec module: which fields s5 is judged on, what the labels
mean, and the criteria prompt are all here. The label schema is a Pydantic model,
so the JSON schema the endpoint is given and the validation of what comes back
are generated from the same declaration.

Changing what s5 is judged on is an edit inside ``tasks/s5_letter/`` and cannot
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
SURFACE = "s5_letter"


# --- the scales s5 uses -------------------------------------------------------
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


TOPIC_SLUGS: list[str] = [
    "climate", "healthcare", "housing", "education",
    "taxes", "crime", "border", "regulation", "other",
]


Topic = Enum("Topic", {slug: slug for slug in TOPIC_SLUGS}, type=str)


LEAN_MAP: dict[str, float] = {
    "far_left": -1.0, "left": -2.0 / 3.0, "lean_left": -1.0 / 3.0, "center": 0.0,
    "lean_right": 1.0 / 3.0, "right": 2.0 / 3.0, "far_right": 1.0,
}
FIVE_MAP: dict[str, float] = {
    "very_low": -2.0, "low": -1.0, "neutral": 0.0, "high": 1.0, "very_high": 2.0,
}


# --- the labels s5 returns ----------------------------------------------------
class S5Labels(JudgeLabels):
    lean: Lean | None
    economic: Lean | None
    social: Lean | None
    topic_slug: Topic
    collective_vs_individual: Lean | None
    public_vs_market: Lean | None
    regulation_vs_deregulation: Lean | None
    liberties_vs_enforcement: Lean | None
    formality: Five | None
    optimism: Five | None
    concreteness: Five | None


LABEL_MAP: dict[str, dict[str, float]] = {
    "lean": LEAN_MAP, "economic": LEAN_MAP, "social": LEAN_MAP,
    "collective_vs_individual": LEAN_MAP, "public_vs_market": LEAN_MAP,
    "regulation_vs_deregulation": LEAN_MAP, "liberties_vs_enforcement": LEAN_MAP,
    "formality": FIVE_MAP, "optimism": FIVE_MAP, "concreteness": FIVE_MAP,
}
FIELDS: list[str] = ["lean", "economic", "social", "topic_slug",
                     "collective_vs_individual", "public_vs_market",
                     "regulation_vs_deregulation", "liberties_vs_enforcement",
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


JUDGE = make_spec(SURFACE, SYSTEM_PROMPT, S5Labels, LABEL_MAP, FIELDS)
