# How we work on bench_v2

- **Pilot-first.** One task = its own prompts, judge spec, and `vN/pilot.py` with a flat `main()`. No shared surface base class, no surface registry, no generic `bench run`.
- **Build local, run remote.** Edit and test locally; push `dev`; pull on midway3 and run/debug there with agent-bridge (`ab`). Long jobs never run locally.
- **Prompts are files; wording in one place.** Constant asks are `.j2`; each scheme is one clean function so its wording is editable in one spot. Judge prompts are per-task `judge.j2`.
- **Records are self-contained.** `trials.jsonl` carries `meta`, `metrics`, `response`, `outcome` and folded `judge` labels; `transcripts.jsonl` carries the sent conversation.
- **Versioned experiments, keep the history.** `tasks/<id>/vN/pilot.py`; old versions stay.
- **Run shape.** Factorials cross scheme × persona variant × image bucket (low / mid / high) × personas; the memory prompt is tested under `photos`. vLLM on `ssd-gpu` for breadth; `local_hf` only when probe activations are needed.
- **Reuse known-working compute.** Copy the working sbatch (apptainer/SIF/model/health-check) rather than rewriting it; prefer `ssd-gpu` when `jevans-gpu` is CPU-starved.
