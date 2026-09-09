#!/bin/bash
#SBATCH --job-name=choice-five
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/probe_choice_five/slurm-%j.out
#SBATCH --error=/project/jevans/tzhang3/dotty-project/linear-political-llm/runs/probe_choice_five/slurm-%j.err
#
# The five forced-choice surfaces: neighbourhood, education, medicine, groceries,
# clothing. 504 generations, no judge calls -- every DV is read by rule.
#
# Serve AND run in one job, unlike probe_voice_serve.sbatch which only serves. The
# client is 504 sequential requests against a server on the same node, so splitting
# them across two jobs would buy nothing and cost a second GPU allocation.
#
# --enable-auto-tool-choice + --tool-call-parser hermes are required for
# s10_groceries. Without them the model emits a perfectly good Hermes-style
# <tool_call>{"name": ...}</tool_call> and the server returns HTTP 400 trying to
# validate that string as JSON -- which is what 72 of 72 groceries trials did.
# Qwen3-VL emits Hermes format; the parser turns it into message.tool_calls.
#
# SIF is pinned, not "newest by mtime". round9_vllm.sbatch derived newest and it bit
# us: a v0.28.0-cu129.sif landed after round 9's data was collected and its api_server
# had dropped --disable-log-requests, so "newest" silently became "broken".
#
set -uo pipefail

REPO=/project/jevans/tzhang3/dotty-project/linear-political-llm
# round9_vllm.sbatch picks the newest vllm-openai-*.sif by mtime. That bit us
# here: a v0.28.0-cu129.sif landed on 2026-09-06 (after round9's data was
# collected) and its api_server dropped --disable-log-requests, so "newest"
# silently became "broken." Pin the SIF round9 actually validated against
# instead of re-deriving "newest" every time a container gets pulled.
SIF=${SIF:-/project/jevans/tzhang3/containers/vllm-openai-v0.11.2.sif}
[ -n "${SIF:-}" ] && [ -f "$SIF" ] || { echo "[sbatch] no vllm SIF found" >&2; exit 1; }
MODELS=$(readlink -f /home/tzhang3/jevans/models 2>/dev/null || echo /home/tzhang3/jevans/models)
MODEL_DIR="$MODELS/Qwen3-VL-8B-Instruct"
[ -d "$MODEL_DIR" ] || { echo "[sbatch] no model dir at $MODEL_DIR" >&2; exit 1; }
SERVED=qwen3-vl-8b-instruct
PORT=${PORT:-8073}
OUT="$REPO/runs/probe_choice_five"

mkdir -p "$OUT"
cd "$REPO"

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

# Write node + port to a well-known file so the login-node side of the probe
# loop can find this job without scraping squeue output by hand.
echo "$(hostname) $PORT" > "$OUT/endpoint.txt"

echo "[sbatch] starting vLLM server on 0.0.0.0:$PORT (host $(hostname))" >&2
"$APPTAINER_BIN" exec --nv \
  --bind "$MODELS:$MODELS:ro" \
  --bind "$REPO:$REPO" \
  --env HF_HUB_OFFLINE=1 \
  "$SIF" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":4}' \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --disable-log-requests \
  > "$OUT/vllm-server-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' EXIT

echo "[sbatch] waiting for /health" >&2
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[sbatch] server ready after ${i}0s" >&2
    echo "ready" >> "$OUT/endpoint.txt"
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[sbatch] server died; see the log" >&2; exit 1; }
  sleep 10
done

# --- the client, on the same node -------------------------------------------
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[sbatch] server never became healthy; not running the client" >&2
  exit 1
fi

export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export VLLM_SERVED_MODEL="$SERVED"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY=${PY:-/home/tzhang3/envs/linear-probe/bin/python}
echo "[sbatch] client: $PY -m bench.pilots.probe_choice_five --phase run ${PHASE_ARGS:---with-agent}" >&2
PHASE_ARGS=${PHASE_ARGS:---with-agent}
"$PY" -m bench.pilots.probe_choice_five --phase run $PHASE_ARGS
RC=$?

echo "[sbatch] client exited rc=$RC" >&2
wc -l "$OUT"/trials.*.jsonl 2>/dev/null || true
kill -TERM "$SERVER_PID" 2>/dev/null || true
exit $RC
