#!/bin/bash
#SBATCH --job-name=benchv2-s16v4-vllm
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --qos=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s16_bias_detect/v4/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s16_bias_detect/v4/slurm-%j.err
#
# bench_v2 s16_bias_detect v4 -- v3 on all 240 statements and all 198 personas.
# One job per scheme, each with its own vLLM server and its own run directory
# (runs/.../v4/<scheme>), so three GPUs share the work and no two processes
# write one store:
#
#   for s in chat agentic agentic_live; do SCHEME=$s sbatch --job-name=s16v4-$s run.sbatch; done
#
# Per scheme: 198 personas x 240 statements x 2 variants = 95040 photo cells,
# plus the item-invariant no_photos baseline (480) -> 95520. Three schemes -> 286560.
# A rerun resumes: cells whose trial_key is already on disk are skipped.
set -uo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
SIF=${SIF:-/project/jevans/tzhang3/containers/vllm-openai-v0.11.2.sif}
MODELS=$(readlink -f /home/tzhang3/jevans/models 2>/dev/null || echo /home/tzhang3/jevans/models)
MODEL_DIR="$MODELS/Qwen3-VL-8B-Instruct"
SERVED=qwen3-vl-8b-instruct
# jobs that land on one node must not share a port
PORT=${PORT:-$((8000 + SLURM_JOB_ID % 1000))}
export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
SCHEME=${SCHEME:?set SCHEME to chat, agentic or agentic_live}
OUT="$REPO/runs/bench_v2/s16_bias_detect/v4/$SCHEME"
ITEMS=${ITEMS:-items/explore_bucket_v1.jsonl}
PER_BUCKET=${PER_BUCKET:-0}
WORKERS=${WORKERS:-32}

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


echo "[sbatch] pilot run: per_bucket=$PER_BUCKET conditions=photos,no_photos scheme=$SCHEME items=$ITEMS workers=$WORKERS" >&2
"$PY" -m bench_v2.tasks.s16_bias_detect.v4.pilot run \
  --adaptor vllm --per-bucket "$PER_BUCKET" \
  --condition photos --condition no_photos \
  --items "$ITEMS" --workers "$WORKERS" --scheme "$SCHEME" --out "$OUT" \
  || { echo "[sbatch] pilot run failed" >&2; exit 1; }

echo "[sbatch] summary" >&2
"$PY" -m bench_v2.tasks.s16_bias_detect.v4.pilot summary --items "$ITEMS" --out "$OUT" > "$OUT/summary.txt" 2>&1 || true
exit 0
