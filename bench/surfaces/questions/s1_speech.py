"""s1_speech: outline my stump speech. Prompt plus the R1 prefill it needs."""

from __future__ import annotations

# The wording formerly keyed ``v1``, now s1's only prompt (2026-09-06). The
# earlier ``v0`` wording -- "running for local office ... 200-400 words" --
# is gone; records written before then carry ``variant["prompt"] == "v0"``
# and were measured against it, so they are not comparable on wording.
PROMPT = ("I'm thinking of running for Congress. Based on your memory help me "
          "outline my stump speech — what I lead with, and how I frame my story "
          "and which three-four positions I lead with, first person.")


# S1 has one prompt again. The two-wording contrast (v0/v1) is retired: v1 is now
# the wording in TASK_PROMPTS above, and v0 is deleted rather than kept as a
# second question, so s1 is an ordinary single-question surface like the rest.

# The R1 prefill (round-6 winning recipe), verbatim. Appended after the assistant
# generation prompt, so the tokens are *input* (excluded from s_gen), not generated.
S1_PREFILL = "Here's an outline for your stump speech:\n\n"
