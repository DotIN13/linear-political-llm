#!/bin/bash
#SBATCH --job-name=face-swap-score
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --qos=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/face_swap_score/slurm-%j.out
#
# Per-token probe scores for the face_swap composites, exactly as EasyPortrait was scored:
# the same prompt (the JSONL is written from generate_portrait_score_data_jsonl.py's
# DEFAULT_PROMPT), the same scorer and the same six probes, Qwen3-VL then Gemma 4.
# Writes results/token_scoring/{qwen3_vl,gemma4}/face_swap/.
#
#   mkdir -p runs/face_swap_score
#   sbatch bench_llm/datasets/face_swap/score.sbatch
set -uo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
cd "$REPO"
export PATH=/home/tzhang3/envs/linear-probe/bin:$PATH
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

echo "[sbatch] partition=${SLURM_JOB_PARTITION:-?} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) rev=$(git rev-parse --short HEAD)" >&2
[ -f data/probes/face_swap_score_data.jsonl ] || python -m bench_llm.datasets.face_swap.build jsonl
DATASETS=face_swap bash scripts/probes/run_token_scoring_qwen3_gemma4.sh || { echo "[sbatch] scoring failed" >&2; exit 1; }
echo "[sbatch] done" >&2
