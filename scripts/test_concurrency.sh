#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

base_url="${BASE_URL:-$(server_base_url)}"
key="$(api_key)"
require_vision_ready_pool "$base_url" "$key"

export BENCHMARK_MAX_NEW_TOKENS="${BENCHMARK_MAX_NEW_TOKENS:-128}"
export BENCHMARK_SAMPLE_INTERVAL="${BENCHMARK_SAMPLE_INTERVAL:-0.5}"
python3 scripts/benchmark_concurrency.py --base-url "$base_url" "$@"
