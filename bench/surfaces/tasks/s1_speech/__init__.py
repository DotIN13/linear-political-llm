"""s1_speech: outline my stump speech.

Two files, for two different reasons.

``prompts/ask.txt`` is the ask, a constant string. The wording formerly keyed ``v1``
is now the only one; the earlier ``v0`` -- "running for local office ... 200-400
words" -- is gone rather than kept as a second question, so records written before
2026-09-06 carry ``variant["prompt"] == "v0"`` and are not comparable on wording.

``prompts/prefill.json`` is the R1 prefill from the round-6 winning recipe. It is json
rather than a ``.txt`` because **it ends in a blank line and that blank line is what
makes it work** -- the model continues an outline instead of starting a turn. In json
that is a visible ``\\n\\n``; at the end of a text file it would be two invisible bytes
that any editor is entitled to eat. It is appended after the assistant generation
prompt, so its tokens are *input* and excluded from ``s_gen``.
"""

from __future__ import annotations

from bench.surfaces.shared.prompts import strings, text

PROMPT = text(__file__)
S1_PREFILL = strings(__file__, "prefill.json")["prefill"]
