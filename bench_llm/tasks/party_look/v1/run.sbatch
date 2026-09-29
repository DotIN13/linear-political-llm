#!/bin/bash
#SBATCH --job-name=bench-llm-party-look
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --qos=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_llm/party_look/v1/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_llm/party_look/v1/slurm-%j.err
#
# bench_llm party_look v1 -- one image, "does it look Democratic or Republican?"
# One job per source, each with its own vLLM server and run directory:
#
#   mkdir -p runs/bench_llm/party_look/v1
#   for s in congress lvis unsplash easyportrait; do SOURCE=$s sbatch --job-name=party-$s bench_llm/tasks/party_look/v1/run.sbatch; done
#
# Per source at the defaults: 10 probe-score strata x 50 images (congress: all 548)
# x 2 orientations of the 1-7 scale. A rerun resumes.
#
# A small pilot of every source in one job (10 strata x 10 = 100 images each, 800 trials):
#
#   SOURCE="congress lvis unsplash easyportrait" PER_STRATUM=10 RUN_NAME=pilot sbatch bench_llm/tasks/party_look/v1/run.sbatch
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
SOURCE=${SOURCE:?set SOURCE to one or more of congress lvis unsplash easyportrait, space-separated}
RUN_NAME=${RUN_NAME:-${SOURCE// /_}}
OUT="$REPO/runs/bench_llm/party_look/v1/$RUN_NAME"
PER_STRATUM=${PER_STRATUM:-$([ "$SOURCE" = congress ] && echo 0 || echo 50)}
SOURCE_ARGS=()
for s in $SOURCE; do SOURCE_ARGS+=(--source "$s"); done
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
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image":1}' \
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


echo "[sbatch] pilot run: source=$SOURCE per_stratum=$PER_STRATUM workers=$WORKERS" >&2
"$PY" -m bench_llm.tasks.party_look.v1.pilot run \
  --adaptor vllm "${SOURCE_ARGS[@]}" --per-stratum "$PER_STRATUM" \
  --workers "$WORKERS" --out "$OUT" \
  || { echo "[sbatch] pilot run failed" >&2; exit 1; }

echo "[sbatch] summary" >&2
"$PY" -m bench_llm.tasks.party_look.v1.pilot summary --out "$OUT" > "$OUT/summary.txt" 2>&1 || true
exit 0
