# How we work on bench_v2

- **Pilot-first.** One task = its own prompts, judge spec, and `v1/pilot.py` with a flat `main()`. No shared surface base class, no surface registry, no generic `bench run`.
- **Build local, run remote.** Edit and test locally; push `dev`; pull on midway3 and run/debug there with agent-bridge (`ab`). Long jobs never run locally.
- **Prompts are files; wording in one place.** Constant asks are `.j2`. Each scheme is one clean function (`_chat_messages` / `_agentic_messages`) so the wording is editable in one spot. Judge prompts are per-task `judge.j2`.
- **Judges are Pydantic, per task.** A task owns its label model and `judge_spec.py`; `bench_v2/judge/` holds only the caller, cache, schema helper and `judge_run`. No shared spec module.
- **Records are self-contained.** `trials.jsonl` carries `meta`, `metrics` (timing/cost/probe), `response`, `outcome`, and folded `judge` labels; `transcripts.jsonl` carries the sent conversation; `conversations/` is content-addressed.
- **Versioned experiments, latest only.** `tasks/<id>/vN/pilot.py`; keep the newest version, not a history.
- **Measurement is hashed.** `measurement_rev` covers adaptors, helpers and prompt material, so wording edits invalidate trials and pilot edits do not.
- **Parity while porting, diverge on purpose.** Ports were byte-identical to `bench` (build + reader). When the instrument is deliberately changed, parity is dropped and the new wording *is* the instrument.
- **Naming.** The persona factor is a `variant` (`bare` / `memory`), not a "clause". Personas are the image items; images come from the sampled dataset, never a hand-built pool.
- **Run shape.** Factorials cross scheme × persona variant × personas; the memory prompt is tested under `photos`. vLLM on `ssd-gpu` (A100/H100) for breadth; `local_hf` only when probe activations are needed.
- **Reuse known-good plumbing.** Copy the working sbatch (apptainer/SIF/model/health-check) rather than rewriting it. Prefer `ssd-gpu` when `jevans-gpu` is CPU-starved.
- **Act, don't ask.** Terse directives mean execute; raise only real ambiguity or a broken instrument.
