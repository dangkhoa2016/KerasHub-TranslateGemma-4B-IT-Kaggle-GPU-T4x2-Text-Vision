#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

base_url="${BASE_URL:-$(server_base_url)}"
source_lang="${SRC_LANG:-English}"
target_lang="${TGT_LANG:-Vietnamese}"
max_tokens="${MAX_NEW_TOKENS:-${DEFAULT_OUTPUT_TOKENS:-256}}"
key="$(api_key)"

headers=(-H "Content-Type: application/json")
if [[ "${API_AUTH_REQUIRED:-true}" != "false" ]]; then
  if [[ -z "$key" ]]; then
    echo "API key is unavailable." >&2
    exit 1
  fi
  headers+=(-H "Authorization: Bearer $key")
fi

require_vision_ready_pool "$base_url" "$key"

if [[ "$#" -eq 0 ]]; then
  images=("$ROOT_DIR/assets/sample-image-with-text.jpg")
else
  images=("$@")
fi

for image_file in "${images[@]}"; do
  if [[ ! -f "$image_file" ]]; then
    echo "Image file not found: $image_file" >&2
    exit 1
  fi

  payload_file="$(mktemp)"
  trap 'rm -f "$payload_file"' EXIT

  python3 - "$image_file" "$source_lang" "$target_lang" "$max_tokens" <<'PY' > "$payload_file"
import base64, json, sys
with open(sys.argv[1], "rb") as handle:
    encoded = base64.b64encode(handle.read()).decode("ascii")
print(json.dumps({
    "image": encoded,
    "source_lang": sys.argv[2],
    "target_lang": sys.argv[3],
    "max_new_tokens": int(sys.argv[4]),
}, ensure_ascii=False))
PY

  echo "--- $source_lang -> $target_lang (image: $image_file)"
  curl -fsS -X POST "$base_url/translate/image" "${headers[@]}" --data @"$payload_file"
  echo
  rm -f "$payload_file"
  trap - EXIT
done
