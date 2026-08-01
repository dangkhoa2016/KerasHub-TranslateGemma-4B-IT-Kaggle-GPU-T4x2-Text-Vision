#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

base_url="${BASE_URL:-$(server_base_url)}"
secret="$(restart_secret)"
if [[ -z "$secret" ]]; then
  echo "Restart secret is unavailable." >&2
  exit 1
fi

curl -fsS -X POST "$base_url/restart" \
  -H "Content-Type: application/json" \
  -H "X-Restart-Secret: $secret" \
  -d "{\"wait_for_jobs\":true,\"timeout\":${RESTART_TIMEOUT:-300}}"
echo
