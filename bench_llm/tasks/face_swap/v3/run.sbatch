#!/bin/bash
#SBATCH --job-name=face-swap
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --qos=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_llm/face_swap/v3/slurm-%x-%j.out
#
# bench_llm face_swap v2 -- backgrounds moved by token splicing, against the v2 pixel composites, both questions and orders:
# text-only headwise probe (top 16 heads) at the last prompt token. No vLLM: the model
# is loaded in transformers so the probe can read it. One model per job:
#
#   data/face_swap_v2/{frames,masks,images} must be present (built by bench_llm.datasets.face_swap.build)
#   mkdir -p runs/bench_llm/face_swap/v3
#   MODEL=qwen3_vl sbatch --job-name=face-swap3-qwen  bench_llm/tasks/face_swap/v3/run.sbatch
#   MODEL=gemma4   sbatch --job-name=face-swap3-gemma bench_llm/tasks/face_swap/v3/run.sbatch
#
# Gemma 4 31B in bf16 needs a GPU with 80 GB or more. A rerun resumes.
set -uo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
MODEL=${MODEL:?set MODEL to qwen3_vl or gemma4}
OUT=${OUT:-"$REPO/runs/bench_llm/face_swap/v3/$MODEL"}   # a smoke test: OUT=... EXTRA="--only normal_lib_p00 --limit 8"

mkdir -p "$OUT"
cd "$REPO"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[sbatch] partition=${SLURM_JOB_PARTITION:-?} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) model=$MODEL rev=$(git rev-parse --short HEAD)" >&2
"$PY" -m bench_llm.tasks.face_swap.v3.pilot run --model "$MODEL" --out "$OUT" ${EXTRA:-} \
  || { echo "[sbatch] run failed" >&2; exit 1; }
"$PY" -m bench_llm.tasks.face_swap.v3.pilot summary --runs "$OUT" > "$OUT/summary.txt" 2>&1 || true
echo "[sbatch] done" >&2
