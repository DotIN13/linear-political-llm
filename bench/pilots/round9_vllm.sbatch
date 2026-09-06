#!/bin/bash
#SBATCH --job-name=r9-vllm
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/pilot_round9v/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/pilot_round9v/slurm-%j.err
#
# Round 9, vLLM path. One job: start the server in apptainer, wait for /health,
# run every trial against it, export, stop the server.
#
# Why one long job rather than the 28-minute chain the HF path needed: the whole
# point of vLLM is that the run is short, and the server takes minutes to load
# weights -- paying that once beats paying it per chunk. If the 4h walltime is
# refused by the queue, lower it; the run itself is expected to be well under it.
set -uo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
# The image name carries its tag: `latest` failed to unpack unprivileged (the
# nsight-compute layer trips umoci's chmod), so a pinned tag is what is on disk.
# Override with SIF=... ; otherwise take the newest vllm-openai-*.sif there is.
SIF=${SIF:-$(ls -t /project/jevans/tzhang3/containers/vllm-openai-*.sif 2>/dev/null | head -1)}
[ -n "${SIF:-}" ] && [ -f "$SIF" ] || { echo "[sbatch] no vllm SIF found" >&2; exit 1; }
MODELS=/home/tzhang3/jevans/models
MODEL_DIR="$MODELS/Qwen3-VL-8B-Instruct"
SERVED=qwen3-vl-8b-instruct
PORT=${PORT:-8073}
OUT="$REPO/runs/pilot_round9v"

mkdir -p "$OUT"
cd "$REPO"
# apptainer is not on PATH by default; it comes from a module, and the module
# name is not stable across versions (singularity/3.9.0, apptainer/1.1.4,
# 1.2.4 and 1.4.1 all exist here). The Aug-3 pull that produced the SIF used
# the absolute path, so prefer that and fall back to the module.
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
echo "[sbatch] apptainer: $APPTAINER_BIN ($("$APPTAINER_BIN" --version 2>&1))" >&2

echo "[sbatch] starting vLLM server on :$PORT" >&2
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
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[sbatch] server ready after ${i}0s" >&2; READY=1; break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[sbatch] server died; see the log" >&2; exit 1; }
  sleep 10
done
[ "${READY:-0}" = 1 ] || { echo "[sbatch] server never became ready" >&2; exit 1; }

export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export VLLM_SERVED_MODEL="$SERVED"
PHASE="${1:-run}"
LIMIT="${2:-0}"
echo "[sbatch] SIF=$SIF  phase=$PHASE  limit=$LIMIT" >&2
"$PY" -m bench.pilots.round9_vllm --phase "$PHASE" --limit "$LIMIT" || exit 1
[ "$PHASE" = run ] && "$PY" -m bench.pilots.round9_vllm --phase export
