#!/bin/bash
#SBATCH --job-name=benchv2-s11-vllm
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s11_health/v1/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/bench_v2/s11_health/v1/slurm-%j.err
#
# bench_v2 s11_health v1 -- vLLM smoke.
#
# One job: start the server in apptainer, wait for /health, run a few trials
# through the bench_v2 pilot, print the summary, stop the server. Copied from
# bench/pilots/round9_vllm.sbatch (the apptainer resolution, the
# /home/tzhang3/jevans symlink fix, HF_HUB_OFFLINE, the pinned SIF) so the two
# paths share one known-good launch.
#
# Port 8000, not round9's 8073: the bench_v2 vllm adaptor defaults to
# http://127.0.0.1:8000/v1 and the pilot passes it no base_url, so the server
# has to be on the port the adaptor already expects.
#
# Partition ssd-gpu, not jevans-gpu. On 2026-09-09 the jevans-gpu submission
# (57993776) sat at Reason=Resources: its only free GPU was on a node with 7
# free CPUs against the 16-CPU request. `gpu` had zero idle GPUs and `ssd-gpu`
# had a fully idle node, so this is the partition that can actually run. The
# directives are only defaults -- override them on the command line if it moves:
#   sbatch --partition=jevans-gpu --account=pi-jevans --qos=jevans-gpu run.sbatch
# No --constraint: the 8B model fits on any GPU here and the node name is logged.
#
# Usage: sbatch run.sbatch [LIMIT] [CELLS]
#   LIMIT=1 -> 1 item x 2 conditions x 2 schemes = 4 cells
#   (no_photos is item-invariant, so that is 2 photo trials + 2 baseline trials)
set -uo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
PY=/home/tzhang3/envs/linear-probe/bin/python
# v0.28.0 dropped --disable-log-requests; pin the version round9 validated.
SIF=${SIF:-/project/jevans/tzhang3/containers/vllm-openai-v0.11.2.sif}
MODELS=$(readlink -f /home/tzhang3/jevans/models 2>/dev/null || echo /home/tzhang3/jevans/models)
MODEL_DIR="$MODELS/Qwen3-VL-8B-Instruct"
SERVED=qwen3-vl-8b-instruct
PORT=${PORT:-8000}
OUT="$REPO/runs/bench_v2/s11_health/v1"
LIMIT=${1:-1}
CELLS=${2:-4}

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

echo "[sbatch] pilot run: limit=$LIMIT cells=$CELLS" >&2
"$PY" -m bench_v2.tasks.s11_health.v1.pilot run \
  --adaptor vllm --limit "$LIMIT" --limit-cells "$CELLS" \
  --items items/explore_bucket_v1.jsonl \
  || { echo "[sbatch] pilot run failed" >&2; exit 1; }

echo "[sbatch] summary" >&2
"$PY" -m bench_v2.tasks.s11_health.v1.pilot summary || true
exit 0
