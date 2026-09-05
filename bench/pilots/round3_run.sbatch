#!/bin/bash
#SBATCH --job-name=r3p-s${STRATUM}
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:28:00
#SBATCH --output=runs/pilot_round3/slurm-%j-s${STRATUM}.out
#SBATCH --error=runs/pilot_round3/slurm-%j-s${STRATUM}.err
# Round-3 pilot, one stratum per job (chained --dependency=afterany; 1 GPU at a time).
set -euo pipefail
REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
cd "$REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} stratum=${STRATUM}"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ "${STRATUM}" = "0" ]; then
  $PY -m bench.pilots.round3_pilot --phases calibrate --n-images 20
fi

$PY -m bench.cli run \
    --items items/pilot_r3_s${STRATUM}.jsonl \
    --surface s1_speech,s4_bonus \
    --conditions C,E \
    --adaptor local_hf \
    --model /home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct \
    --probe combined_ideology_headwise_linear \
    --seed 42 \
    --no-candidate-gate \
    --out runs/pilot_round3 \
    --conversations conversations

echo "=== done stratum ${STRATUM}: $(date) ==="
wc -l runs/pilot_round3/trials.jsonl
