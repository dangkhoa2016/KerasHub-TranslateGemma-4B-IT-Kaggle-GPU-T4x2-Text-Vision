#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

pid_file="$ROOT_DIR/state/server.pid"
if old_pid="$(read_managed_pid "$pid_file" "src/server.py" 2>/dev/null)"; then
  echo "Server is already running (PID $old_pid)."
  exit 0
fi

nohup python3 -u src/server.py > log/server.stdout.log 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$pid_file"
echo "Started API coordinator (PID $pid) on port $PORT."
echo "Liveness: $(server_base_url)/health/live"
echo "Readiness: $(server_base_url)/health/ready"
echo "Logs: log/server.log, log/server.stdout.log, log/worker-*.log"
