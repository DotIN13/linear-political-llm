"""The s17_neutralize v1 judge spec -- owned by the version.

The scale is the paper's own: ``scripts/data_gen/label_rewrites.py`` labelled the
rewrites LEFT / LEAN_LEFT / NEUTRAL / LEAN_RIGHT / RIGHT with gpt-5, and Figure 3
is the distribution of those five labels. It is kept, lower-cased, so a bench_v2
rewrite and a paper rewrite read on one scale. ``coherent`` is there because the
paper's rightward steering "harms coherence" (Table 2), and a garbled rewrite
should be countable rather than silently scored as neutral.

The judge sees only the rewrite, and is not told it is one: told "this was meant
to be neutral", a rater drifts toward neutral.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from bench_v2.helpers.prompts import render
from bench_v2.judge import DEFAULT_JUDGE_MODEL, JudgeLabels, JudgeSpec, caps_for

TASK_DIR = Path(__file__).resolve().parent
RUBRIC = "s17_neutralize"


class Lean5(str, Enum):
    left = "left"
    lean_left = "lean_left"
    neutral = "neutral"
    lean_right = "lean_right"
    right = "right"


LEAN5_MAP: dict[str, float] = {
    "left": -1.0, "lean_left": -0.5, "neutral": 0.0, "lean_right": 0.5, "right": 1.0,
}


class S17Labels(JudgeLabels):
    lean: Lean5 | None
    coherent: bool | None


LABEL_MAP: dict[str, dict[str, float]] = {"lean": LEAN5_MAP}
FIELDS: list[str] = ["lean", "coherent"]
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


JUDGE = make_spec(RUBRIC, SYSTEM_PROMPT, S17Labels, LABEL_MAP, FIELDS)
