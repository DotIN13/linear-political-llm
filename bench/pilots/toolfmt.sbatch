#!/bin/bash
#SBATCH --job-name=toolfmt-pilot
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=runs/pilot_toolfmt/slurm-%j.out
#SBATCH --error=runs/pilot_toolfmt/slurm-%j.err

# One model per job, one GPU, <= 30 min (docs/bench/08).
#   MODEL=qwen  -> Q1/Q2 render + Q3 behaviour (4 arms x 20 items) + Q4 probe
#   MODEL=gemma -> Q1/Q2 + Q3 only (the ideology probe is fitted on Qwen)
set -euo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  model=${MODEL:-qwen}"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ "${MODEL:-qwen}" = "gemma" ]; then
    $PY -m bench.pilots.tool_format_probe \
        --phases template,behavior \
        --model-path /home/tzhang3/jevans/models/gemma-4-E4B-it \
        --model-family gemma4 --tag gemma4-e4b --system-in-user
else
    $PY -m bench.pilots.tool_format_probe \
        --phases template,behavior,probe \
        --model-path /home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct \
        --model-family qwen3-vl --tag qwen3-vl-8b \
        --probe combined_ideology_headwise_linear \
        --probe-modes M1_user,M2_tool
fi

echo "=== done: $(date) ==="
ls -la runs/pilot_toolfmt/
