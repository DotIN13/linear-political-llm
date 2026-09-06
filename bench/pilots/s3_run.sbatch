#!/bin/bash
#SBATCH --job-name=s3-pilot
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/pilot_s3/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/pilot_s3/slurm-%j.err
# S3 pilot (docs/bench/13): 35 generations x 600 tokens, 1 H200, <30 min.
#   C  = 9 items x (chat x2 reps + agentic x1)
#   E  = 2 schemes x 4 seeds
# Usage: sbatch bench/pilots/s3_run.sbatch
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

$PY -m bench.pilots.s3_pilot --phase run

echo "=== done: $(date) ==="
wc -l runs/pilot_s3/trials.jsonl 2>/dev/null || true
