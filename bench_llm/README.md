# bench_llm

A standalone harness for generation benchmarks: show a model a stimulus, ask it one
thing, read the answer. It imports nothing from bench_v2 (a test checks this).

The package is plumbing plus helpers. Everything that makes a benchmark what it is
(which images, which labels, the wording, how a reply becomes a number, what the
summary compares) lives in the task's `pilot.py` and its `.j2` files.

| module | provides |
|---|---|
| `types.py` | `Item`, `Conversation`, `Trial`, `Response`, `Outcome` |
| `adaptors.py` | `OpenAICompat`: greedy chat completions with first-token logprobs, for a local vLLM server or a hosted API |
| `run.py` | `cells`, `run_cells` (resume by trial key, threads, fsynced `trials.jsonl`, `manifest.json`), `read_trials`, `instrument_rev` |
| `sources.py` | `read_csv`, `dedupe`, `stratify`, `sample_per_stratum`, `read_table`, `to_items`, `resolve`, `file_sha16` |
| `prompts.py` | `render` a `.j2`, `user_turn`, `conversation` |
| `readers.py` | `option_probs`, `top_option`, `expected_value`, `log_odds`, `first_mention`, `matches`, `word_count` |
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

## Adding a task

Make `tasks/<id>/v1/` with its `.j2` asks and a `pilot.py` that defines `build(item,
variant) -> Trial` and `read(response, trial) -> Outcome` from the helpers, then calls
`run.run_cells`. Keep both pure so they are testable without a server.
