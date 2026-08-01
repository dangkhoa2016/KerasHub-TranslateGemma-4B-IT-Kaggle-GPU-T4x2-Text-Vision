#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

pid_file="$ROOT_DIR/state/server.pid"
if ! pid="$(read_managed_pid "$pid_file" "src/server.py" 2>/dev/null)"; then
  echo "Server is not running (stale/invalid PID file removed if present)."
  exit 0
fi

kill -TERM "$pid"
for _ in $(seq 1 "${STOP_WAIT_SECONDS:-330}"); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "Server stopped."
    exit 0
  fi
  sleep 1
done

echo "Graceful stop timed out; sending SIGKILL." >&2
kill -KILL "$pid" 2>/dev/null || true
rm -f "$pid_file"
