#!/usr/bin/env bash
set -euo pipefail

# Run token scoring for both Qwen3-VL and Gemma4 across all generated JSONL datasets.
#
# Usage:
#   bash scripts/probes/run_token_scoring_qwen3_gemma4.sh
#
# Override defaults via environment variables:
#   DATASETS="congress news"   (subset of dataset names to run)
#   PROBE_SPECS="combined_ideology:headwise_linear combined_ideology:layerwise_linear combined_ideology:layerwise_rfm" \
#   TOP_K=16 \
#   DTYPE=auto \
#   DEVICE_MAP=auto \
#   OVERWRITE=1 \
#   bash scripts/probes/run_token_scoring_qwen3_gemma4.sh
#
# You can also pass extra CLI flags directly (forwarded to token_scoring.py):
#   bash scripts/probes/run_token_scoring_qwen3_gemma4.sh --limit 500

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCORER="$ROOT_DIR/scripts/probes/token_scoring.py"

# Shared defaults
MODE="${MODE:-vision}"
TOP_K="${TOP_K:-16}"
DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DATA_DIR="${DATA_DIR:-results/probes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/token_scoring}"
HAS_IMAGES="${HAS_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"

# Probe selection
PROBE_SPECS="${PROBE_SPECS:-combined_ideology:headwise_linear combined_ideology:layerwise_linear combined_ideology:layerwise_rfm}"

# Model registry
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/home/tzhang3/jevans/models/Qwen3-VL-8B-Instruct}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-/project/jevans/tzhang3/models/gemma-4-31B-it}"

# Dataset registry: name -> jsonl path (relative to ROOT_DIR)
declare -A DATASET_PATHS=(
  [congress]="data/probes/congress_score_data.jsonl"
  [news]="data/probes/news_images_score_data.jsonl"
  [twitter]="data/probes/twitter_images_score_data.jsonl"
  [unsplash]="data/probes/unsplash25k_score_data.jsonl"
  [red_blue]="data/probes/red_blue_score_data.jsonl"
  [maga_hat]="data/probes/maga_hat_score_data.jsonl"
  [lvis]="data/probes/lvis_score_data.jsonl"
)

# Allow caller to restrict which datasets to run (space-separated names)
if [[ -n "${DATASETS:-}" ]]; then
  read -ra DATASET_NAMES <<< "$DATASETS"
else
  DATASET_NAMES=(congress news twitter unsplash red_blue maga_hat)
fi

EXTRA_ARGS=("$@")

mkdir -p "$OUTPUT_ROOT"

read -ra probe_specs_arr <<< "$PROBE_SPECS"
if [[ ${#probe_specs_arr[@]} -eq 0 ]]; then
  echo "PROBE_SPECS is empty. Provide at least one PREFIX:PROBE spec." >&2
  exit 1
fi

run_model_dataset() {
  local model_name="$1"
  local model_path="$2"
  local model_family="$3"
  local dataset_name="$4"
  local data_path="$5"
  local output_dir="$OUTPUT_ROOT/$model_name/$dataset_name"

  mkdir -p "$output_dir"

  echo
  echo "=== Running token scoring: $model_name / $dataset_name ==="
  echo "  model_path : $model_path"
  echo "  model_family: $model_family"
  echo "  data       : $data_path"
  echo "  output_dir : $output_dir"

  local extra_args=(
    --mode "$MODE"
    --data "$data_path"
    --top-k "$TOP_K"
    --dtype "$DTYPE"
    --device-map "$DEVICE_MAP"
    --data-dir "$DATA_DIR"
  )

  if [[ "$HAS_IMAGES" == "1" ]]; then
    extra_args+=(--has-images)
  fi

  if [[ "$OVERWRITE" == "1" ]]; then
    extra_args+=(--overwrite)
  fi

  python "$SCORER" \
    --model-path "$model_path" \
    --model-family "$model_family" \
    --output-dir "$output_dir" \
    --probe-specs "${probe_specs_arr[@]}" \
    "${extra_args[@]}" \
    "${EXTRA_ARGS[@]}"
}

for dataset_name in "${DATASET_NAMES[@]}"; do
  data_path="${DATASET_PATHS[$dataset_name]:-}"
  if [[ -z "$data_path" ]]; then
    echo "Unknown dataset: $dataset_name (skipping)" >&2
    continue
  fi

  run_model_dataset "qwen3_vl" "$QWEN_MODEL_PATH" "qwen3-vl" "$dataset_name" "$data_path"
  run_model_dataset "gemma4"   "$GEMMA4_MODEL_PATH" "gemma4"  "$dataset_name" "$data_path"
done

echo
echo "Token scoring complete for all datasets."
