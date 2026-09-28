#!/bin/bash
#SBATCH --job-name=benchv2-s16v3-vllm
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --qos=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s16_bias_detect/v3/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s16_bias_detect/v3/slurm-%j.err
#
# bench_v2 s16_bias_detect v3 -- v2 with the numbering reversed (1 conservative .. 5 liberal); the paper's bias detection (Findings of EMNLP
# 2025, section 3.2.1) under the full scheme x persona factorial: the model says
# whether a lawmaker's statement is liberal, conservative or neutral.
#
# The reading is deterministic (the first label in the reply), so there is no
# judge phase. Same launch as s7_family_chat v2 (apptainer resolution, /health
# wait, pinned SIF), so the two share one known-good path.
#
# Usage: PER_BUCKET=10 sbatch run.sbatch
#   PER_BUCKET=10 -> 30 personas x 18 statements x 3 schemes x 2 variants = 3240
#   photo cells, plus the item-invariant no_photos baseline (108) -> 3348 total.
#   PER_BUCKET=0 runs every item (198 personas -> 21384 + 108).
#   SCHEMES=agentic_live sbatch run.sbatch   narrows the scheme set.
set -uo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
SIF=${SIF:-/project/jevans/tzhang3/containers/vllm-openai-v0.11.2.sif}
MODELS=$(readlink -f /home/tzhang3/jevans/models 2>/dev/null || echo /home/tzhang3/jevans/models)
MODEL_DIR="$MODELS/Qwen3-VL-8B-Instruct"
SERVED=qwen3-vl-8b-instruct
PORT=${PORT:-8000}
OUT="$REPO/runs/bench_v2/s16_bias_detect/v3"
ITEMS=${ITEMS:-items/explore_bucket_v1.jsonl}
PER_BUCKET=${PER_BUCKET:-10}
WORKERS=${WORKERS:-16}
SCHEMES=${SCHEMES:-}

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

SCHEME_ARGS=()
for s in ${SCHEMES//,/ }; do SCHEME_ARGS+=(--scheme "$s"); done

echo "[sbatch] pilot run: per_bucket=$PER_BUCKET conditions=photos,no_photos schemes=${SCHEMES:-all} items=$ITEMS workers=$WORKERS" >&2
"$PY" -m bench_v2.tasks.s16_bias_detect.v3.pilot run \
  --adaptor vllm --per-bucket "$PER_BUCKET" \
  --condition photos --condition no_photos \
  --items "$ITEMS" --workers "$WORKERS" "${SCHEME_ARGS[@]}" \
  || { echo "[sbatch] pilot run failed" >&2; exit 1; }

echo "[sbatch] summary" >&2
"$PY" -m bench_v2.tasks.s16_bias_detect.v3.pilot summary --items "$ITEMS" || true
exit 0
