#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

curl -fsS "$(server_base_url)/health/live" || true
echo
key="$(api_key)"
if [[ -n "$key" ]]; then
  curl -sS "$(server_base_url)/health/ready?details=1" \
    -H "Authorization: Bearer $key" || true
else
  curl -sS "$(server_base_url)/health/ready" || true
fi
echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
fi
