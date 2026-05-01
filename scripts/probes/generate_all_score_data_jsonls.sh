#!/usr/bin/env bash
set -euo pipefail

# Generate all score-data JSONLs in one command.
#
# Example:
#   bash scripts/probes/generate_all_score_data_jsonls.sh \
#     --output-dir data/probes \
#     --limit 1000 \
#     --recursive \
#     --news-image-dir data/news_images \
#     --twitter-image-dir data/twitter_images \
#     --unsplash-image-dir data/unsplash
#
# Optional prompt overrides:
#   --congress-prompt "..." --news-prompt "..." --twitter-prompt "..." --unsplash-prompt "..." --red-blue-prompt "..." --maga-hat-prompt "..."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBES_DIR="$ROOT_DIR/scripts/probes"

OUTPUT_DIR="data/probes"
LIMIT=""
RECURSIVE=0

CONGRESS_IMAGE_DIR="data/congress_images"
CONGRESS_HS_PATH="data/HS116_members.csv"
CONGRESS_CUR_PATH="data/legislators-current.json"
CONGRESS_HIST_PATH="data/legislators-historical.json"
CONGRESS_RESIZED_IMAGE_WIDTH="250"

NEWS_IMAGE_DIR="data/news_images"
NEWS_MAX_WIDTH="800"

TWITTER_IMAGE_DIR="data/twitter_images"
TWITTER_MAX_WIDTH="800"

UNSPLASH_IMAGE_DIR="data/unsplash"
RED_BLUE_IMAGE_DIR="data/gemini_tie_pairs/red_blue"
MAGA_HAT_IMAGE_DIR="data/gemini_tie_pairs/maga_hat_balanced/images"

CONGRESS_PROMPT=""
NEWS_PROMPT=""
TWITTER_PROMPT=""
UNSPLASH_PROMPT=""
RED_BLUE_PROMPT=""
MAGA_HAT_PROMPT=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Shared options:
  --output-dir PATH                Output directory for all JSONLs (default: $OUTPUT_DIR)
  --limit N                        Optional limit passed to all generators
  --recursive                      Enable recursive image search for news/twitter/unsplash

Dataset path options:
  --congress-image-dir PATH
  --congress-hs-path PATH
  --congress-current-legislators-path PATH
  --congress-historical-legislators-path PATH
  --congress-resized-image-width N

  --news-image-dir PATH
  --news-max-width N

  --twitter-image-dir PATH
  --twitter-max-width N

  --unsplash-image-dir PATH
  --maga-hat-image-dir PATH

Prompt overrides:
  --congress-prompt TEXT
  --news-prompt TEXT
  --twitter-prompt TEXT
  --unsplash-prompt TEXT
  --red-blue-prompt TEXT
  --maga-hat-prompt TEXT

Additional dataset path options:
  --red-blue-image-dir PATH
  --maga-hat-image-dir PATH

Outputs:
  <output-dir>/congress_score_data.jsonl
  <output-dir>/news_images_score_data.jsonl
  <output-dir>/twitter_images_score_data.jsonl
  <output-dir>/unsplash25k_score_data.jsonl
  <output-dir>/red_blue_score_data.jsonl
  <output-dir>/maga_hat_score_data.jsonl
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --recursive)
      RECURSIVE=1; shift ;;

    --congress-image-dir)
      CONGRESS_IMAGE_DIR="$2"; shift 2 ;;
    --congress-hs-path)
      CONGRESS_HS_PATH="$2"; shift 2 ;;
    --congress-current-legislators-path)
      CONGRESS_CUR_PATH="$2"; shift 2 ;;
    --congress-historical-legislators-path)
      CONGRESS_HIST_PATH="$2"; shift 2 ;;
    --congress-resized-image-width)
      CONGRESS_RESIZED_IMAGE_WIDTH="$2"; shift 2 ;;

    --news-image-dir)
      NEWS_IMAGE_DIR="$2"; shift 2 ;;
    --news-max-width)
      NEWS_MAX_WIDTH="$2"; shift 2 ;;

    --twitter-image-dir)
      TWITTER_IMAGE_DIR="$2"; shift 2 ;;
    --twitter-max-width)
      TWITTER_MAX_WIDTH="$2"; shift 2 ;;

    --unsplash-image-dir)
      UNSPLASH_IMAGE_DIR="$2"; shift 2 ;;
    --red-blue-image-dir)
      RED_BLUE_IMAGE_DIR="$2"; shift 2 ;;
    --maga-hat-image-dir)
      MAGA_HAT_IMAGE_DIR="$2"; shift 2 ;;

    --congress-prompt)
      CONGRESS_PROMPT="$2"; shift 2 ;;
    --news-prompt)
      NEWS_PROMPT="$2"; shift 2 ;;
    --twitter-prompt)
      TWITTER_PROMPT="$2"; shift 2 ;;
    --unsplash-prompt)
      UNSPLASH_PROMPT="$2"; shift 2 ;;
    --red-blue-prompt)
      RED_BLUE_PROMPT="$2"; shift 2 ;;
    --maga-hat-prompt)
      MAGA_HAT_PROMPT="$2"; shift 2 ;;

    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

COMMON_LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  COMMON_LIMIT_ARGS=(--limit "$LIMIT")
fi

RECURSIVE_ARGS=()
if [[ "$RECURSIVE" -eq 1 ]]; then
  RECURSIVE_ARGS=(--recursive)
fi

CONGRESS_PROMPT_ARGS=()
if [[ -n "$CONGRESS_PROMPT" ]]; then
  CONGRESS_PROMPT_ARGS=(--prompt "$CONGRESS_PROMPT")
fi

NEWS_PROMPT_ARGS=()
if [[ -n "$NEWS_PROMPT" ]]; then
  NEWS_PROMPT_ARGS=(--prompt "$NEWS_PROMPT")
fi

TWITTER_PROMPT_ARGS=()
if [[ -n "$TWITTER_PROMPT" ]]; then
  TWITTER_PROMPT_ARGS=(--prompt "$TWITTER_PROMPT")
fi

UNSPLASH_PROMPT_ARGS=()
if [[ -n "$UNSPLASH_PROMPT" ]]; then
  UNSPLASH_PROMPT_ARGS=(--prompt "$UNSPLASH_PROMPT")
fi

RED_BLUE_PROMPT_ARGS=()
if [[ -n "$RED_BLUE_PROMPT" ]]; then
  RED_BLUE_PROMPT_ARGS=(--prompt "$RED_BLUE_PROMPT")
fi

MAGA_HAT_PROMPT_ARGS=()
if [[ -n "$MAGA_HAT_PROMPT" ]]; then
  MAGA_HAT_PROMPT_ARGS=(--prompt "$MAGA_HAT_PROMPT")
fi

run_if_missing() {
  local output_path="$1"
  shift

  if [[ -f "$output_path" ]]; then
    echo "Skipping existing output: $output_path"
    return 0
  fi

  "$@"
}

echo "[1/6] Generating Congress JSONL"
run_if_missing "$OUTPUT_DIR/congress_score_data.jsonl" \
  python "$PROBES_DIR/generate_congress_score_data_jsonl.py" \
    --image-dir "$CONGRESS_IMAGE_DIR" \
    --hs-path "$CONGRESS_HS_PATH" \
    --current-legislators-path "$CONGRESS_CUR_PATH" \
    --historical-legislators-path "$CONGRESS_HIST_PATH" \
    --resized-image-width "$CONGRESS_RESIZED_IMAGE_WIDTH" \
    --output-path "$OUTPUT_DIR/congress_score_data.jsonl" \
    "${COMMON_LIMIT_ARGS[@]}" \
    "${CONGRESS_PROMPT_ARGS[@]}"

echo "[2/6] Generating News JSONL"
run_if_missing "$OUTPUT_DIR/news_images_score_data.jsonl" \
  python "$PROBES_DIR/generate_news_score_data_jsonl.py" \
    --image-dir "$NEWS_IMAGE_DIR" \
    --max-width "$NEWS_MAX_WIDTH" \
    --output-path "$OUTPUT_DIR/news_images_score_data.jsonl" \
    "${COMMON_LIMIT_ARGS[@]}" \
    "${RECURSIVE_ARGS[@]}" \
    "${NEWS_PROMPT_ARGS[@]}"

echo "[3/6] Generating Twitter JSONL"
run_if_missing "$OUTPUT_DIR/twitter_images_score_data.jsonl" \
  python "$PROBES_DIR/generate_twitter_score_data_jsonl.py" \
    --image-dir "$TWITTER_IMAGE_DIR" \
    --max-width "$TWITTER_MAX_WIDTH" \
    --output-path "$OUTPUT_DIR/twitter_images_score_data.jsonl" \
    "${COMMON_LIMIT_ARGS[@]}" \
    "${RECURSIVE_ARGS[@]}" \
    "${TWITTER_PROMPT_ARGS[@]}"

echo "[4/5] Generating Unsplash JSONL"
run_if_missing "$OUTPUT_DIR/unsplash25k_score_data.jsonl" \
  python "$PROBES_DIR/generate_unsplash_25k_score_data_jsonl.py" \
    --image-dir "$UNSPLASH_IMAGE_DIR" \
    --output-path "$OUTPUT_DIR/unsplash25k_score_data.jsonl" \
    "${COMMON_LIMIT_ARGS[@]}" \
    "${RECURSIVE_ARGS[@]}" \
    "${UNSPLASH_PROMPT_ARGS[@]}"

echo "[5/5] Generating Red-Blue JSONL"
run_if_missing "$OUTPUT_DIR/red_blue_score_data.jsonl" \
  python "$PROBES_DIR/generate_red_blue_score_data_jsonl.py" \
    --image-dir "$RED_BLUE_IMAGE_DIR" \
    --output-path "$OUTPUT_DIR/red_blue_score_data.jsonl" \
    "${COMMON_LIMIT_ARGS[@]}" \
    "${RECURSIVE_ARGS[@]}" \
    "${RED_BLUE_PROMPT_ARGS[@]}"

echo "[6/6] Generating MAGA-Hat JSONL"
run_if_missing "$OUTPUT_DIR/maga_hat_score_data.jsonl" \
  python "$PROBES_DIR/generate_red_blue_score_data_jsonl.py" \
    --image-dir "$MAGA_HAT_IMAGE_DIR" \
    --output-path "$OUTPUT_DIR/maga_hat_score_data.jsonl" \
    --id-prefix "maga_hat" \
    --source "gemini_tie_pairs_maga_hat" \
    "${COMMON_LIMIT_ARGS[@]}" \
    "${RECURSIVE_ARGS[@]}" \
    "${MAGA_HAT_PROMPT_ARGS[@]}"

echo "Done. JSONLs written to: $OUTPUT_DIR"
