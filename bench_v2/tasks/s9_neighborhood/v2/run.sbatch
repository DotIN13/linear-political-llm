#!/bin/bash
#SBATCH --job-name=benchv2-s9v2-vllm
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s9_neighborhood/v2/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s9_neighborhood/v2/slurm-%j.err
#
# bench_v2 s9_neighborhood v2 -- vLLM smoke.
#
# Same launch as v1, copied rather than rewritten: apptainer resolution, the
# /home/tzhang3/jevans symlink fix, HF_HUB_OFFLINE, the pinned SIF, port 8000
# because the bench_v2 vllm adaptor defaults to it, and ssd-gpu because that is
# the partition with an idle node. Only the version, the module path and the
# item selection changed.
#
# Directives are defaults; override on the command line if it moves:
#   sbatch --partition=jevans-gpu --account=pi-jevans --qos=jevans-gpu run.sbatch
#
# Slurm opens the stdout file before the script body runs, so the run directory
# has to exist BEFORE submitting or the job dies on start:
#
#   mkdir -p /project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s9_neighborhood/v2
#   sbatch bench_v2/tasks/s9_neighborhood/v2/run.sbatch 1 24
#
# Usage: sbatch run.sbatch [PER_BUCKET] [CELLS]
#   PER_BUCKET=1 -> 1 item per image-lean bucket = 3 items, x 2 conditions
#                   x 6 variants (3 schemes x bare/memory) = 36 cells, of which
#                   6 are the item-invariant no_photos baseline.
#   PER_BUCKET=0 takes --limit items from the head of the file instead, which is
#   all one bucket: the file is grouped, so use 0 only for a smoke.
set -uo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
SIF=${SIF:-/project/jevans/tzhang3/containers/vllm-openai-v0.11.2.sif}
MODELS=$(readlink -f /home/tzhang3/jevans/models 2>/dev/null || echo /home/tzhang3/jevans/models)
MODEL_DIR="$MODELS/Qwen3-VL-8B-Instruct"
SERVED=qwen3-vl-8b-instruct
PORT=${PORT:-8000}
OUT="$REPO/runs/bench_v2/s9_neighborhood/v2"
PER_BUCKET=${1:-1}
CELLS=${2:-36}

[ -f "$SIF" ] || { echo "[sbatch] no SIF at $SIF" >&2; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "[sbatch] no model at $MODEL_DIR" >&2; exit 1; }

mkdir -p "$OUT"
cd "$REPO"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

APPTAINER_BIN=${APPTAINER_BIN:-}
if [ -z "$APPTAINER_BIN" ]; then
  for cand in /software/apptainer-1.4.1-el8-x86_64/bin/apptainer \
              /software/apptainer-*/bin/apptainer; do
    [ -x "$cand" ] && { APPTAINER_BIN=$cand; break; }
  done
fi
if [ -z "$APPTAINER_BIN" ]; then
  command -v apptainer >/dev/null 2>&1 || module load apptainer 2>/dev/null || true
  APPTAINER_BIN=$(command -v apptainer || true)
fi
[ -n "$APPTAINER_BIN" ] || { echo "[sbatch] no apptainer binary found" >&2; exit 1; }

echo "[sbatch] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >&2
echo "[sbatch] apptainer: $APPTAINER_BIN ($("$APPTAINER_BIN" --version 2>&1))" >&2

echo "[sbatch] starting vLLM on 127.0.0.1:$PORT" >&2
"$APPTAINER_BIN" exec --nv \
  --bind "$MODELS:$MODELS:ro" \
  --bind "$REPO:$REPO" \
  --env HF_HUB_OFFLINE=1 \
  "$SIF" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED" \
    --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":4}' \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests \
  > "$OUT/vllm-server-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' EXIT

echo "[sbatch] waiting for /health" >&2
READY=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[sbatch] server ready after ~${i}0s" >&2
    READY=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || {
    echo "[sbatch] server died; see $OUT/vllm-server-$SLURM_JOB_ID.log" >&2
    exit 1
  }
  sleep 10
done
[ "$READY" = 1 ] || { echo "[sbatch] server never became ready" >&2; exit 1; }

echo "[sbatch] pilot run: per_bucket=$PER_BUCKET cells=$CELLS" >&2
"$PY" -m bench_v2.tasks.s9_neighborhood.v2.pilot run \
  --adaptor vllm --per-bucket "$PER_BUCKET" --limit-cells "$CELLS" \
  --items items/explore_bucket_v1.jsonl --workers 16 \
  || { echo "[sbatch] pilot run failed" >&2; exit 1; }

echo "[sbatch] summary" >&2
"$PY" -m bench_v2.tasks.s9_neighborhood.v2.pilot summary || true
exit 0
