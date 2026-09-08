#!/bin/bash
#SBATCH --job-name=r9-six
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/pilot_round9/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/pilot_round9/slurm-%j.err
# Round-9: six surfaces x (chat|agentic) x prefill arms + S3 order balance.
# 318 generations, split into 4 serial jobs (--dependency=afterany):
#   chunk 0 -> calibrate + A chat s1,s2,s3            (72 gen + 40 calib forwards)
#   chunk 1 -> A chat s4,s5,s6 + A agentic s1,s2       (90 gen)
#   chunk 2 -> A agentic s3,s4,s5,s6                   (90 gen)
#   chunk 3 -> P (s1 chat prefill on) + C baseline     (66 gen)
# Usage: sbatch bench/pilots/round9_run.sbatch "<phases>" "<chunk>"
set -euo pipefail
REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
PHASES="${1:-calibrate,run}"
CHUNK="${2:-0}"
cd "$REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} phases=${PHASES} chunk=${CHUNK}"
nvidia-smi --query-gpu=name,memory.total --format=csv

$PY -m bench.pilots.round9_pilot --phase "$PHASES" --chunk "$CHUNK"

echo "=== done: $(date) ==="
wc -l runs/pilot_round9/trials.jsonl 2>/dev/null || true
