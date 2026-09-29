"""bench_llm: a standalone harness for generation benchmarks.

Show a model a stimulus, ask it something, read the answer. The package provides
the plumbing and small helpers; a task's ``pilot.py`` assembles them:

* ``types``     -- Item, Conversation, Trial, Response, Outcome
* ``adaptors``  -- an OpenAI-compatible chat client (a local vLLM server or a hosted API)
* ``run``       -- the resumable run loop, the trial key and the instrument hash
* ``sources``   -- helpers to read score tables, join labels, stratify and sample
* ``prompts``   -- helpers to render a task's ``.j2`` asks and build a user turn
* ``readers``   -- helpers to read a reply: option probabilities, expected value,
  log-odds, first keyword named, a regex in the head
* ``stats``     -- Spearman, Pearson, AUC, stdlib only

It imports nothing from bench_v2.
"""
