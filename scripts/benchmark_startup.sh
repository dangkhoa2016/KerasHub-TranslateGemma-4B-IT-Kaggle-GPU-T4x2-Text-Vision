#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env
if [[ "${RUN_STARTUP_BENCHMARK:-0}" != "1" ]]; then
  echo "Refusing long startup benchmark unless RUN_STARTUP_BENCHMARK=1" >&2
  exit 2
fi
exec python3 scripts/benchmark_startup.py
