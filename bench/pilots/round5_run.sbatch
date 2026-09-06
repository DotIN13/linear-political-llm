#!/bin/bash
#SBATCH --job-name=r5-s1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=runs/pilot_round5/slurm-%j.out
#SBATCH --error=runs/pilot_round5/slurm-%j.err
# Round-5: new S1 prompt (no word cap, 1200 tokens), 4 arms / 34 generations.
# Submitted as 3 serial jobs (--dependency=afterany):
#   1. calibrate + A-chat      -> sbatch ... "calibrate,run" "A" "chat"
#   2. A-agentic               -> sbatch ... "run" "A" "agentic"
#   3. C + M                   -> sbatch ... "run" "C,M" ""
# Usage: sbatch bench/pilots/round5_run.sbatch "<phases>" "<arms>" "<schemes>"
set -euo pipefail
REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
PHASES="${1:-calibrate,run}"
ARMS="${2:-}"
SCHEMES="${3:-}"
cd "$REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} phases=${PHASES} arms=${ARMS} schemes=${SCHEMES}"
nvidia-smi --query-gpu=name,memory.total --format=csv

ARGS=(--phases "$PHASES" --n-images 20)
if [ -n "$ARMS" ]; then ARGS+=(--arms "$ARMS"); fi
if [ -n "$SCHEMES" ]; then ARGS+=(--schemes "$SCHEMES"); fi

$PY -m bench.pilots.round5_pilot "${ARGS[@]}"

echo "=== done: $(date) ==="
wc -l runs/pilot_round5/trials.jsonl 2>/dev/null || true
