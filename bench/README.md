# bench

Stimulus -> surface -> adaptor -> store. Design specs live in `docs/bench/` (six files);
they are the authority, this file is only how to run it.

```bash
PY=/home/tzhang3/envs/linear-probe/bin/python   # torch 2.11, patched transformers 5.6.0.dev0

$PY -m bench.cli surfaces                                    # 8 surfaces + their requires
$PY -m bench.cli adaptors                                    # capability matrix
$PY -m bench.cli check --surface vote2020 --adaptor opencode # gate, before you queue

# stimuli (CPU, ~2 min; first call reduces datasets/lvis/lvis_v1_train.json to a cache)
$PY -m bench.cli sample --bins 10 --per-bin 400 --images-per-item 3 \
      --split explore,confirm --out items/

# the only step that touches a GPU
sbatch bench/smoke.sbatch          # 20 items x {vote2020,tea_coffee} x {C,E} = 80 trials

$PY -m bench.cli score --run runs/smoke
$PY -m pytest bench/tests -q       # 100 tests, no GPU
```

Conventions that matter:

- **Resume is not a flag.** `RunStore` dedups on
  `trial_key = sha256(surface, item_id, condition, adaptor, model, seed, code_rev)`, so
  rerunning any `bench run` command is always safe. `--resume` exists only as a no-op.
- **Items are write-once.** `bench sample` refuses to overwrite `items/*.jsonl`. New
  filters mean a new filename, otherwise old runs stop meaning anything.
- **Conversations live in `conversations/<sha2>/<sha>.json`**; a trial row carries only
  `conversation_sha`, so `trials.jsonl` stays greppable.
- `runs/`, `items/`, `conversations/`, `judge_cache/` are gitignored. Nothing here is committed.

Not implemented this round: `judge` and `report` are stubs; `sample` implements filters
F1–F3 and F5–F6 but **not F4** (OCR against a political wordlist — needs an OCR
dependency this repo does not have); surfaces are the eight two-alternative choices only
(no scale / ranking / senate / speech yet); one question wording per surface, no rephrasings.
