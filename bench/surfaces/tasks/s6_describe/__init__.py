"""s6_describe: describe me to someone who has never met me.

``_POLITICS_WORDS`` / ``extract_mentions_politics`` is this task's own reader -- the DV
is whether politics came up *unprompted*, which is only a question here.

The reader is worth reading before trusting it: the word list is loose enough that
``mentions_politics`` fires on 780 of the 1,890 real responses in the golden corpus,
including plenty that are not about politics. Reported, not changed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from bench.judges.specs import judge_specs
from bench.surfaces.shared.prompts import text
from bench.surfaces.shared.surface import GenerationSurface
from bench.types import Trial

PROMPT = text(__file__)


# **Two nets, on purpose.** The DV here is "did politics come up *unprompted*",
# and how wide you cast decides what that sentence means.
#
# `_POLITICS_WORDS` is the original and is **deliberately unchanged**: it is what
# every s6 number reported so far was measured with, and silently redefining a DV
# that has already been reported is worse than a loose one. It is loose, though --
# it contains `government`, `policy`, `party`, `vote` and `citizen`, so all four of
# these are True:
#
#   "We had a party last weekend."      "I vote for pizza."
#   "My policy is to always be kind."   "A thoughtful citizen of the world."
#
# Over the 1,890-row golden corpus it fires on 780. On s6's own judged round it
# fired 0/44, so on real s6 output the looseness has cost nothing yet -- the risk
# is forward-looking.
#
# `_POLITICS_WORDS_STRICT` drops every term with a common non-political sense. A
# hit here is hard to explain except as politics.
_POLITICS_WORDS = [
    "politic", "liberal", "conservative", "democrat", "republican", "left-wing",
    "right-wing", "leftwing", "rightwing", "election", "voting", "vote", "ideology",
    "ideological", "civic", "citizen", "activist", "progressive", "moderate",
    "government", "policy", "protest", "candidate", "party",
]

# Dropped as ambiguous: government, policy, party, vote, voting, citizen, civic,
# moderate, progressive, candidate, activist, protest. Each has an everyday sense
# that has nothing to do with politics.
_POLITICS_WORDS_STRICT = [
    "politic", "liberal", "conservative", "democrat", "republican", "left-wing",
    "right-wing", "leftwing", "rightwing", "election", "ideology", "ideological",
]


def _hits(text: str, words: Sequence[str]) -> List[str]:
    lowered = (text or "").lower()
    return [w for w in words if w in lowered]


def extract_mentions_politics(text: str) -> bool:
    """The original, loose net. Unchanged -- see the note above."""
    return bool(_hits(text, _POLITICS_WORDS))


def extract_mentions_politics_strict(text: str) -> bool:
    """Only terms with no common non-political sense."""
    return bool(_hits(text, _POLITICS_WORDS_STRICT))


class _S6Surface(GenerationSurface):
    name = "s6_describe"
    family = "generation"
    prompt = PROMPT
    judge_spec = judge_specs().get("s6_describe")
    max_new_tokens = 600          # "a short paragraph", plus whatever preamble

    def _deterministic(self, text: str, trial: Optional[Trial]) -> Dict[str, Any]:
        mentions = extract_mentions_politics(text)
        # `primary` stays on the loose net so the DV that has been reported keeps
        # meaning what it meant. The strict reading rides alongside, and the words
        # that fired are recorded so a disagreement can be read rather than guessed.
        return {"primary": 1.0 if mentions else 0.0,
                "mentions_politics": mentions,
                "mentions_politics_strict": extract_mentions_politics_strict(text),
                "politics_terms": _hits(text, _POLITICS_WORDS)}
