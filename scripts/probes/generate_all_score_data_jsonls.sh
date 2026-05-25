#!/usr/bin/env bash
set -euo pipefail

# Generate all score-data JSONLs with fixed settings.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBES_DIR="$ROOT_DIR/scripts/probes"

OUTPUT_DIR="data/probes"
if [[ $# -ne 0 ]]; then
  echo "This script does not accept arguments. Edit variables in the file if needed." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

run_if_missing() {
  local output_path="$1"
  shift

  if [[ -f "$output_path" ]]; then
    echo "Skipping existing output: $output_path"
    return 0
  fi

  "$@"
}

echo "[1/8] Generating Congress JSONL"
run_if_missing "$OUTPUT_DIR/congress_score_data.jsonl" \
  python "$PROBES_DIR/generate_congress_score_data_jsonl.py" \
    --image-dir "data/congress_images" \
    --hs-path "data/HS116_members.csv" \
    --current-legislators-path "data/legislators-current.json" \
    --historical-legislators-path "data/legislators-historical.json" \
    --resized-image-width "250" \
    --output-path "$OUTPUT_DIR/congress_score_data.jsonl"

echo "[2/8] Generating News JSONL"
run_if_missing "$OUTPUT_DIR/news_images_score_data.jsonl" \
  python "$PROBES_DIR/generate_news_score_data_jsonl.py" \
    --image-dir "data/news_images" \
    --max-width "800" \
    --output-path "$OUTPUT_DIR/news_images_score_data.jsonl" \
    --recursive

echo "[3/8] Generating Twitter JSONL"
run_if_missing "$OUTPUT_DIR/twitter_images_score_data.jsonl" \
  python "$PROBES_DIR/generate_twitter_score_data_jsonl.py" \
    --image-dir "data/twitter_images" \
    --max-width "800" \
    --output-path "$OUTPUT_DIR/twitter_images_score_data.jsonl" \
    --recursive

echo "[4/8] Generating Unsplash JSONL"
run_if_missing "$OUTPUT_DIR/unsplash25k_score_data.jsonl" \
  python "$PROBES_DIR/generate_unsplash_25k_score_data_jsonl.py" \
    --image-dir "data/unsplash" \
    --output-path "$OUTPUT_DIR/unsplash25k_score_data.jsonl" \
    --recursive

echo "[5/8] Generating Red-Blue JSONL"
run_if_missing "$OUTPUT_DIR/red_blue_score_data.jsonl" \
  python "$PROBES_DIR/generate_red_blue_score_data_jsonl.py" \
    --image-dir "data/gemini_tie_pairs/red_blue" \
    --output-path "$OUTPUT_DIR/red_blue_score_data.jsonl"

echo "[6/8] Generating MAGA-Hat JSONL"
run_if_missing "$OUTPUT_DIR/maga_hat_score_data.jsonl" \
  python "$PROBES_DIR/generate_red_blue_score_data_jsonl.py" \
    --image-dir "data/gemini_tie_pairs/maga_hat_balanced/images" \
    --output-path "$OUTPUT_DIR/maga_hat_score_data.jsonl" \
    --id-prefix "maga_hat" \
    --source "gemini_tie_pairs_maga_hat"

echo "[7/8] Generating Protest-Sign JSONL"
run_if_missing "$OUTPUT_DIR/protest_sign_score_data.jsonl" \
  python "$PROBES_DIR/generate_red_blue_score_data_jsonl.py" \
    --image-dir "results/gemini_tie_pairs/protest_sign/images" \
    --output-path "$OUTPUT_DIR/protest_sign_score_data.jsonl" \
    --id-prefix "protest_sign" \
    --source "gemini_tie_pairs_protest_sign"

echo "[8/8] Generating EasyPortrait JSONL"
run_if_missing "$OUTPUT_DIR/easyportrait_score_data.jsonl" \
  python "$PROBES_DIR/generate_portrait_score_data_jsonl.py" \
    --image-dir "datasets/EasyPortrait/data/images" \
    --max-width "800" \
    --output-path "$OUTPUT_DIR/easyportrait_score_data.jsonl" \
    --recursive

echo "Done. JSONLs written to: $OUTPUT_DIR"
