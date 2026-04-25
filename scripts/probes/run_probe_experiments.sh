#!/usr/bin/env bash
# Train textual and combined ideology probes for 3 models × 3 probe types.
# Runs 18 combinations sequentially, one model is loaded per probe+direction pair.
#
# Usage (from repo root):
#   bash scripts/run_probe_experiments.sh
#
# Override defaults via environment variables:
#   TEXTUAL_DATA=data/probes/textual_ideology.jsonl \
#   COMBINED_DATA=data/probes/combined_ideology.jsonl \
#   MODELS="qwen3-vl gemma4" \
#   PROBES="headwise_linear layerwise_rfm" \
#   bash scripts/run_probe_experiments.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------

TEXTUAL_DATA="${TEXTUAL_DATA:-data/probes/textual_ideology.jsonl}"
COMBINED_DATA="${COMBINED_DATA:-data/probes/combined_ideology.jsonl}"
DATA_DIR="${DATA_DIR:-results/probes}"
DTYPE="${DTYPE:-bfloat16}"
LIMIT="${LIMIT:-}"                  # leave empty to use all samples
EXTRA_ARGS="${EXTRA_ARGS:-}"        # any extra flags to pass through

MODELS="${MODELS:-qwen3-vl gemma4 llama-3.2-vision}"
PROBES="${PROBES:-headwise_linear layerwise_linear layerwise_rfm}"

# ---------------------------------------------------------------------------
# Model registry: path and mode for each model key
# ---------------------------------------------------------------------------

declare -A MODEL_PATHS=(
    [qwen3-vl]="/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct"
    [gemma4]="/project/jevans/tzhang3/models/gemma-4-31B-it"
    [llama-3.2-vision]="/project/jevans/tzhang3/models/Llama-3.2-11B-Vision-Instruct"
)

declare -A MODEL_MODES=(
    [qwen3-vl]="qwen3-vl"
    [gemma4]="vision"
    [llama-3.2-vision]="vision"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

maybe_limit() {
    if [[ -n "${LIMIT:-}" ]]; then
        echo "--limit $LIMIT"
    fi
}

run_probe() {
    local model="$1"
    local probe="$2"
    local prefix="$3"
    local data="$4"

    local model_path="${MODEL_PATHS[$model]}"
    local mode="${MODEL_MODES[$model]}"

    echo
    echo "=== ${model}  |  ${probe}  |  ${prefix} ==="

    # shellcheck disable=SC2046
    python -m probes.cli \
        --model-path  "$model_path" \
        --mode        "$mode" \
        --probe       "$probe" \
        --prefix      "$prefix" \
        --data        "$data" \
        --data-dir    "$DATA_DIR" \
        --dtype       "$DTYPE" \
        $(maybe_limit) \
        ${EXTRA_ARGS}
}

# ---------------------------------------------------------------------------
# Main loop: model × probe × direction
# ---------------------------------------------------------------------------

read -ra _MODELS <<< "$MODELS"
read -ra _PROBES <<< "$PROBES"

for model in "${_MODELS[@]}"; do
    for probe in "${_PROBES[@]}"; do
        run_probe "$model" "$probe" "textual_ideology" "$TEXTUAL_DATA"
        run_probe "$model" "$probe" "combined_ideology" "$COMBINED_DATA"
    done
done

echo
echo "All probe experiments complete."
