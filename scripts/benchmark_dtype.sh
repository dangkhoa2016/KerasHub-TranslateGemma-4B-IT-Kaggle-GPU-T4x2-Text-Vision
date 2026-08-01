#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

if [[ "${RUN_DTYPE_BENCHMARK:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
This benchmark performs multiple full model reloads and is intentionally opt-in.
Run it with:
  RUN_DTYPE_BENCHMARK=1 bash scripts/benchmark_dtype.sh
It does NOT edit .env. MODEL_DTYPE_OVERRIDE is inherited only by each benchmark server process.
EOF
  exit 2
fi

wait_ready() {
  local timeout="${DTYPE_READY_TIMEOUT:-1800}"
  for _ in $(seq 1 "$timeout"); do
    if curl -fsS "$(server_base_url)/health/ready?all=1" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

run_dtype() {
  local dtype="$1"
  echo "=== dtype benchmark: $dtype ==="
  bash scripts/stop.sh || true
  MODEL_DTYPE_OVERRIDE="$dtype" bash scripts/start.sh
  if ! wait_ready; then
    echo "Worker pool did not become ready for dtype=$dtype" >&2
    return 1
  fi
  local rc=0
  BENCHMARK_LABEL="dtype-$dtype" \
    BENCHMARK_MAX_NEW_TOKENS="${BENCHMARK_MAX_NEW_TOKENS:-128}" \
    bash scripts/test_concurrency.sh || rc=$?
  bash scripts/stop.sh || true
  return "$rc"
}

status=0
run_dtype bfloat16 || status=$?
run_dtype float16 || status=$?

echo "=== restoring stable .env dtype (${MODEL_DTYPE:-bfloat16}) ==="
bash scripts/stop.sh || true
bash scripts/start.sh
wait_ready || {
  echo "Stable server did not return to ready state." >&2
  exit 1
}

echo "Dtype benchmark logs are under log/benchmark-dtype-*.json and .csv"
exit "$status"
