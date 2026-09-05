#!/bin/bash
#SBATCH --job-name=bench-smoke2
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=runs/smoke2/slurm-%j.out
#SBATCH --error=runs/smoke2/slurm-%j.err

# Round-2 smoke. 20 items (2 per stratum, from the v2 sample)
#   x 2 surfaces (vote2020, tea_coffee)
#   x 2 conditions (C, E)
#   x 2 A/B orders
# The naive count is 20*2*2*2 = 160, but condition E is item-invariant: its
# conversation carries no image, so it is run once per (surface, order) instead
# of once per item. That removes (20-1)*2*2 = 76 duplicate trials:
#   160 - 76 = 84  =  80 image trials (C) + 4 baseline trials (E)
set -euo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
cd "$REPO"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 20 items, 2 per stratum, so the score table spans the whole gradient.
if [ ! -f items/smoke20_v2.jsonl ]; then
  $PY - <<'PYEOF'
import json
from collections import defaultdict
per_stratum = defaultdict(list)
for line in open("items/explore_v2.jsonl", encoding="utf-8"):
    row = json.loads(line)
    per_stratum[row["stratum"]].append(row)
picked = [row for s in sorted(per_stratum) for row in per_stratum[s][:2]]
with open("items/smoke20_v2.jsonl", "w", encoding="utf-8") as handle:
    for row in picked:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
print(f"smoke set: {len(picked)} items over strata {sorted(per_stratum)}")
PYEOF
fi

MODEL=/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct

# --- gate 1: the acceptance command, with the synthetic no-image prompt -------
echo "=================== bench check --surface vote2020 --adaptor local_hf ==================="
$PY -m bench.cli check --surface vote2020 --adaptor local_hf --model "$MODEL" || true

# --- gate 2: the same check on a real 3-image stimulus, both surfaces ---------
echo "=================== bench check (both surfaces, real item, condition C) ==================="
$PY -m bench.cli check --surface vote2020,tea_coffee --adaptor local_hf --model "$MODEL" \
    --items items/smoke20_v2.jsonl --condition C || true

echo "=================== bench run --out runs/smoke2 ==================="
$PY -m bench.cli run \
    --items items/smoke20_v2.jsonl \
    --surface vote2020,tea_coffee \
    --conditions C,E \
    --adaptor local_hf \
    --model "$MODEL" \
    --probe combined_ideology_headwise_linear \
    --seed 42 \
    --out runs/smoke2 \
    --conversations conversations

# Identical command a second time. Every trial_key is already in the store, so
# this must do zero forward passes -- resume is the default, not a flag.
echo "=================== second invocation (resume check) ==================="
$PY -m bench.cli run \
    --items items/smoke20_v2.jsonl \
    --surface vote2020,tea_coffee \
    --conditions C,E \
    --adaptor local_hf \
    --model "$MODEL" \
    --probe combined_ideology_headwise_linear \
    --seed 42 \
    --out runs/smoke2 \
    --conversations conversations

echo "=================== bench score --run runs/smoke2 ==================="
$PY -m bench.cli score --run runs/smoke2

echo "=================== bench score --by variant (diagnostic) ==================="
$PY -m bench.cli score --run runs/smoke2 --by variant

echo "=================== trials.jsonl line count ==================="
wc -l runs/smoke2/trials.jsonl
