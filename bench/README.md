# bench

Stimulus -> surface -> adaptor -> store. Design specs live in `docs/bench/`;
01–06 are the authority, **07 is the diff** (six design changes made in round 2).
This file is only how to run it.

```bash
PY=/home/tzhang3/envs/linear-probe/bin/python   # torch 2.11, patched transformers 5.6.0.dev0

$PY -m bench.cli surfaces                        # 8 surfaces, their options and variants
$PY -m bench.cli adaptors                        # capability matrix

# gate, before you queue. Two gates in one command:
#   capability -- can this adaptor produce what this surface needs?
#   candidate  -- are "A"/"B" single tokens, and does the model answer with one?
# The candidate gate does a real forward pass, so it loads the model. --no-candidates skips it.
$PY -m bench.cli check --surface vote2020 --adaptor local_hf
$PY -m bench.cli check --surface all --adaptor opencode --no-candidates

# stimuli (CPU, ~3 min; first call reduces datasets/lvis/lvis_v1_train.json to a cache)
$PY -m bench.cli sample --bins 10 --per-bin 400 --images-per-item 3 \
      --split explore,confirm --suffix v2 --out items/

# the only step that touches a GPU
sbatch bench/smoke2.sbatch        # 20 items x {vote2020,tea_coffee} x {C,E} x {ab,ba} = 84 trials

$PY -m bench.cli score --run runs/smoke2
$PY -m bench.cli score --run runs/smoke2 --by variant          # diagnostic split
$PY -m bench.cli score --run runs/smoke2 --filter variant.order=ab
$PY -m pytest bench/tests -q      # 160 tests, no GPU
```

Conventions that matter:

- **The answer is a letter, not a word.** Each surface carries `options` (two
  semantic choices) and three `phrasings`; the question renders as `A. <opt> /
  B. <opt> / Answer with a single letter.` and the outcome is
  `logP("A") - logP("B")`, re-oriented onto `options[0]`. Word candidates were
  asymmetric under the tokenizer (`"Biden"` -> `["B","iden"]` vs `"Trump"`).
- **Both A/B orders always run.** `order="ba"` is negated before averaging;
  their difference is stored as `position_bias`.
- **Variants are repeated measures.** `bench score` averages over them *before*
  counting, so `n` is items, never rows. Both `n_items` and `n_rows` are printed.
- **Resume is not a flag.** `RunStore` dedups on
  `trial_key = sha256(surface, item_id, condition, variant, adaptor, model, seed, measurement_rev)`.
- **`measurement_rev`, not git HEAD.** It hashes `bench/adaptors/**`,
  `bench/surfaces/**`, `bench/types.py`, `bench/store.py` and the probe weights.
  Editing this README or `bench/cli.py` does not invalidate a single trial.
  `code_rev` is still recorded in every row and in the manifest -- as provenance.
- **The stratum is the independent variable**, not `image_mean_mean`; items say so
  in `primary_iv`. `bench score` prints the `stratum -> s_img` monotonicity check.
- **Condition E is run once.** Its conversation has no image and is byte-identical
  for every item, so it is stored as `item_id="__baseline__"` and broadcast at
  score time as `outcome_minus_baseline`.
- **Sampling filters on integrity only.** `no_person`, `no_text_cats`, `objects:`,
  `aspect:`, `tokens:` still exist behind `--filters` as sensitivity analyses;
  by default the same quantities ride along as annotations
  (`n_persons`, `has_text_cat`, `n_objects`, `aspect`, `num_image_tokens`).
- **Items are write-once.** `bench sample --suffix v2` writes `items/explore_v2.jsonl`;
  it refuses to overwrite an existing stimulus file.
- **Conversations live in `conversations/<sha2>/<sha>.json`**; a trial row carries only
  `conversation_sha`, so `trials.jsonl` stays greppable.
- `runs/`, `items/`, `conversations/`, `judge_cache/` are gitignored.

Known limits, so nobody rediscovers them:

- `n_persons` is a LVIS *annotation* count, and LVIS is federated: `person` is
  annotated in 1,928 of 100,170 images (1.9%). `n_persons == 0` means "not
  annotated", not "no person in frame". Controlling the face channel properly
  needs a detector.
- `judge` and `report` are still stubs; `sample` still does not do F4 (OCR against
  a political wordlist -- needs an OCR dependency this repo does not have).
- Surfaces are the eight two-alternative choices only (no scale / ranking /
  senate / speech yet). Three phrasings exist per surface but only phrasing 0 is
  in the declared variant space this round: `active_phrasings = [0]`.
