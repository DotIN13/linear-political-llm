#!/bin/bash
#SBATCH --job-name=bench-smoke
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=runs/smoke/slurm-%j.out
#SBATCH --error=runs/smoke/slurm-%j.err

# Smoke run: 20 items x 2 surfaces (vote2020, tea_coffee) x 2 conditions (C, E)
# = 80 trials, one prefill each. Resumable: rerunning skips finished trial_keys.
set -euo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
cd "$REPO"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 20 items, 2 per decile, so the score table actually spans the gradient.
if [ ! -f items/smoke20.jsonl ]; then
  $PY - <<'PYEOF'
import json
from collections import defaultdict
per_decile = defaultdict(list)
for line in open("items/explore.jsonl", encoding="utf-8"):
    row = json.loads(line)
    per_decile[row["decile"]].append(row)
picked = [row for d in sorted(per_decile) for row in per_decile[d][:2]]
with open("items/smoke20.jsonl", "w", encoding="utf-8") as handle:
    for row in picked:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
print(f"smoke set: {len(picked)} items over deciles {sorted(per_decile)}")
PYEOF
fi

$PY -m bench.cli run \
    --items items/smoke20.jsonl \
    --surface vote2020,tea_coffee \
    --conditions C,E \
    --adaptor local_hf \
    --model /home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct \
    --probe combined_ideology_headwise_linear \
    --seed 42 \
    --out runs/smoke \
    --conversations conversations

# Run the identical command a second time. Every trial_key is already in the
# store, so this must do zero forward passes -- resume is the default, not a flag.
echo "=================== second invocation (resume check) ==================="
$PY -m bench.cli run \
    --items items/smoke20.jsonl \
    --surface vote2020,tea_coffee \
    --conditions C,E \
    --adaptor local_hf \
    --model /home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct \
    --probe combined_ideology_headwise_linear \
    --seed 42 \
    --out runs/smoke \
    --conversations conversations

echo "=================== bench score --run runs/smoke ==================="
$PY -m bench.cli score --run runs/smoke
