#!/bin/bash
#SBATCH --job-name=r4-buckets
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=runs/pilot_round4/slurm-%j.out
#SBATCH --error=runs/pilot_round4/slurm-%j.err
# Round-4 pilot: calibration (20 images x 2 schemes) + 22 generations (arms A/B/C).
# One GPU, one job; ~15 min total, well under the 30-minute bound.
set -euo pipefail
REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
cd "$REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv

$PY -m bench.pilots.round4_pilot --phases calibrate,run --n-images 20

echo "=== done: $(date) ==="
wc -l runs/pilot_round4/trials.jsonl
