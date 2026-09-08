"""s2_proposal: what should my neighborhood association do with $50,000.

A constant ask and nothing else -- no reader of its own, no subclass, no prefill. So
the entry file only says where the wording is, and ``prompts/ask.txt`` is the wording.
"""

from __future__ import annotations

from bench.surfaces.shared.prompts import text

PROMPT = text(__file__)
