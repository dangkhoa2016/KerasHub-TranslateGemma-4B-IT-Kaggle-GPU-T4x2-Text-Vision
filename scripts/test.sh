#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

base_url="${BASE_URL:-$(server_base_url)}"
input_file="${1:-$ROOT_DIR/data/input.txt}"
[[ -f "$input_file" ]] || input_file="$ROOT_DIR/data/input.example.txt"
key="$(api_key)"
max_tokens="${MAX_NEW_TOKENS:-${DEFAULT_OUTPUT_TOKENS:-256}}"

headers=(-H "Content-Type: application/json")
if [[ "${API_AUTH_REQUIRED:-true}" != "false" ]]; then
  if [[ -z "$key" ]]; then
    echo "API key is unavailable." >&2
    exit 1
  fi
  headers+=(-H "Authorization: Bearer $key")
fi

# Text inference must use the same full multimodal workers as the vision test.
# This prevents a false PASS from a text-only model that can answer /translate.
require_vision_ready_pool "$base_url" "$key"

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  src="English"
  tgt="Vietnamese"
  text="$line"
  if [[ "$line" == *"|"*"|"* ]]; then
    src="${line%%|*}"
    rest="${line#*|}"
    tgt="${rest%%|*}"
    text="${rest#*|}"
  fi
  payload="$(python3 - "$src" "$tgt" "$text" "$max_tokens" <<'PY'
import json, sys
print(json.dumps({
    "source_lang": sys.argv[1],
    "target_lang": sys.argv[2],
    "text": sys.argv[3],
    "max_new_tokens": int(sys.argv[4]),
}, ensure_ascii=False))
PY
)"
  echo "--- $src -> $tgt"
  curl -fsS -X POST "$base_url/translate" "${headers[@]}" -d "$payload"
  echo
done < "$input_file"
