#!/bin/bash
#SBATCH --job-name=r8-s1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/pilot_round8/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/pilot_round8/slurm-%j.err
# Round-8: prefill-as-variant, S1 x prompt x scheme. 88 generations x 1400 tokens,
# split into 3 serial jobs (--dependency=afterany):
#   chunk 0 -> calibrate + A(v0,chat)             (18 gen)
#   chunk 1 -> A(v0,agentic) + A(v1,chat)          (36 gen)
#   chunk 2 -> A(v1,agentic) + C(all 4 cells)      (34 gen)
# Usage: sbatch bench/pilots/round8_run.sbatch "<phases>" "<chunk>"
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

$PY -m bench.pilots.round8_pilot --phase "$PHASES" --chunk "$CHUNK"

echo "=== done: $(date) ==="
wc -l runs/pilot_round8/trials.jsonl 2>/dev/null || true
