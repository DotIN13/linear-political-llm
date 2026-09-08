"""s6_describe: describe me to someone who has never met me.

``_POLITICS_WORDS`` / ``extract_mentions_politics`` is this task's own reader -- the DV
is whether politics came up *unprompted*, which is only a question here.

The reader is worth reading before trusting it: the word list is loose enough that
``mentions_politics`` fires on 780 of the 1,890 real responses in the golden corpus,
including plenty that are not about politics. Reported, not changed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from bench.judges.specs import judge_specs
from bench.surfaces.shared.prompts import text
from bench.surfaces.shared.surface import GenerationSurface
from bench.types import Trial

PROMPT = text(__file__)


_POLITICS_WORDS = [
    "politic", "liberal", "conservative", "democrat", "republican", "left-wing",
    "right-wing", "leftwing", "rightwing", "election", "voting", "vote", "ideology",
    "ideological", "civic", "citizen", "activist", "progressive", "moderate",
    "government", "policy", "protest", "candidate", "party",
]


def extract_mentions_politics(text: str) -> bool:
    lowered = (text or "").lower()
    return any(w in lowered for w in _POLITICS_WORDS)


class _S6Surface(GenerationSurface):
    name = "s6_describe"
    family = "generation"
    prompt = PROMPT
    judge_spec = judge_specs().get("s6_describe")
    max_new_tokens = 600          # "a short paragraph", plus whatever preamble

    def _deterministic(self, text: str, trial: Optional[Trial]) -> Dict[str, Any]:
        mentions = extract_mentions_politics(text)
        return {"primary": 1.0 if mentions else 0.0, "mentions_politics": mentions}
