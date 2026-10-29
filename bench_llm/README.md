# bench_llm

A standalone harness for generation benchmarks: show a model a stimulus, ask it one
thing, read the answer. It imports nothing from bench_v2 (a test checks this).

The package is plumbing plus helpers. Everything that makes a benchmark what it is
(which images, which labels, the wording, how a reply becomes a number, what the
summary compares) lives in the task's `pilot.py` and its `.j2` files.

| module | provides |
|---|---|
| `types.py` | `Item`, `Conversation`, `Trial`, `Response`, `Outcome` |
| `adaptors.py` | `OpenAICompat`: greedy chat completions with first-token logprobs, for a local vLLM server or a hosted API; `HFProbe`: the model in transformers, greedy, plus one trained probe's score at the last prompt token |
| `run.py` | `cells`, `run_cells` (resume by trial key, threads, fsynced `trials.jsonl`, `manifest.json`), `read_trials`, `instrument_rev` |
| `sources.py` | `read_csv`, `dedupe`, `stratify`, `sample_per_stratum`, `read_table`, `to_items`, `resolve`, `file_sha16` |
| `prompts.py` | `render` a `.j2`, `user_turn`, `conversation` |
| `readers.py` | `option_probs`, `top_option`, `expected_value`, `log_odds`, `first_mention`, `matches`, `word_count` |
| `splice.py` | `VisionTap` (read or override a model's image tokens at the encoder output, deepstack included), `splice_plan` and `apply_plan`: splice image tokens between two equally framed images; `ranked_cells` picks the cells for a best-token swap |
| `token_scores.py` | `ImageTokenScorer`: per-image-token probe scores during a trial, by the token scoring script's own functions |
| `stats.py` | `mean_se`, `spearman`, `pearson`, `corr_se`, `auc` |

Each row of `trials.jsonl` carries the item's data, the variant, the messages sent,
the response and the outcome, so a summary never needs another file. The trial key
hashes the task, item, variant, adaptor, model, seed and `instrument_rev`: the
content of the package modules and of every task's prompt files. Pilots are not
hashed, so editing a summary never invalidates a trial; editing an ask does.

## Tasks

**`party_look/v1`**: does this image look Democratic or Republican? One image from
LVIS, Unsplash, EasyPortrait or the congressional portraits, placed on a 1 to 7 scale,
asked with Democrats at 1 and with Republicans at 1. Each record keeps the digit the
model wrote, the first token's log-probability of every digit 1 to 7 (and the top 20 as
returned), and the probability-weighted score, all on −1 Democratic to +1 Republican.
The summary sets each reading against the image's probe score (Spearman) and, for
Congress, against the member's party (AUC) and DW-NOMINATE score.

```bash
python -m bench_llm.tasks.party_look.v1.pilot plan --source congress --per-stratum 0
mkdir -p runs/bench_llm/party_look/v1
for s in congress lvis unsplash easyportrait; do
  SOURCE=$s sbatch --job-name=party-$s bench_llm/tasks/party_look/v1/run.sbatch
done
python -m bench_llm.tasks.party_look.v1.pilot summary \
  --runs runs/bench_llm/party_look/v1/congress --runs runs/bench_llm/party_look/v1/lvis
```

The probe was trained partly on portraits, so on the congressional portraits the party,
not the probe, is the ground truth.

**`face_lean/v1`**: the 20 faces in `faces.csv` (12 Congress members, 8 EasyPortrait),
asked "liberal or conservative leaning? Answer with one word." Each record keeps the word
and the text-only headwise probe (`textual_ideology_headwise_linear`, top 16 heads) at the
last prompt token. Runs through `HFProbe`, one model per job, no vLLM.

```bash
MODEL=qwen3_vl sbatch bench_llm/tasks/face_lean/v1/run.sbatch
MODEL=gemma4   sbatch bench_llm/tasks/face_lean/v1/run.sbatch
```

**`face_latte/v1`**: the same 20 faces, asked "Take a guess, do you think I like latte or
coffee? Answer with one word." Same probe and position as `face_lean`; its export joins
each face with its `face_lean` record.

```bash
MODEL=qwen3_vl sbatch bench_llm/tasks/face_latte/v1/run.sbatch
MODEL=gemma4   sbatch bench_llm/tasks/face_latte/v1/run.sbatch
```

**`face_latte/v2`**: v1's question asked both ways, "latte or coffee" and "coffee or
latte", 40 trials per model; the summary counts the faces that get the same answer in both.

```bash
MODEL=qwen3_vl sbatch bench_llm/tasks/face_latte/v2/run.sbatch
MODEL=gemma4   sbatch bench_llm/tasks/face_latte/v2/run.sbatch
```

**`face_latte/v3`**: v2 plus each candidate word scored by teacher forcing (`HFProbe`
`candidates`), so every trial also keeps the log-odds of coffee over latte (above 0 leans coffee, as the probe leans conservative), spellings pooled.

```bash
MODEL=qwen3_vl sbatch bench_llm/tasks/face_latte/v3/run.sbatch
MODEL=gemma4   sbatch bench_llm/tasks/face_latte/v3/run.sbatch
```

**`face_swap/v1`**: the politics and latte questions, both orders, on the 110 background-swap
composites from `bench_llm/datasets/face_swap` (440 trials per model): the word, the text-only
probe at the last prompt token, and whole-word log-odds (conservative - liberal, coffee - latte).
Each trial scores only its own question's words (`Trial.meta["candidates"]`).

```bash
MODEL=qwen3_vl sbatch --job-name=face-swap-qwen  bench_llm/tasks/face_swap/v1/run.sbatch
MODEL=gemma4   sbatch --job-name=face-swap-gemma bench_llm/tasks/face_swap/v1/run.sbatch
```

**`face_swap/v2`**: the background moved by token splicing (`splice.py`): person P2's photo
runs with P1's background tokens in its background cells, and the mean of P1's background
tokens where P1's own person sat; against a fill-only check, P2's normal run, and v2's pixel
composites. 1,240 trials per model.

**`face_swap/v3`**: the best-token swap. The 20 most liberal and 20 most conservative faces
(by their own token scores), aligned on the face, are paired; a share (2 to 100%) of the cells
that are person in both takes the other face's tokens, either the ones the donor's own
image-token probe ranks most extreme (`token_scores.py`) or as many at random. Both
directions, 2,080 trials per model.

**`face_swap/v4`**: v3 across the whole portrait: every cell of the frame can move, not only
the cells that are person in both photos. 2,080 trials per model.

## Adding a task

Make `tasks/<id>/v1/` with its `.j2` asks and a `pilot.py` that defines `build(item,
variant) -> Trial` and `read(response, trial) -> Outcome` from the helpers, then calls
`run.run_cells`. Keep both pure so they are testable without a server.
