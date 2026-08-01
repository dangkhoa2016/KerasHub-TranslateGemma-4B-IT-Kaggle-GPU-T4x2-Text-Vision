#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

if [[ "${_FULL_RESTART_BG:-0}" != "1" ]]; then
  mkdir -p "$ROOT_DIR/log"
  nohup env _FULL_RESTART_BG=1 bash "$(realpath "$0")" "$@" \
    > "$ROOT_DIR/log/full_restart.stdout.log" 2>&1 &
  echo "full_restart.sh started in background (PID $!). Log: log/full_restart.stdout.log"
  exit 0
fi

"$ROOT_DIR/scripts/stop_tunnel.sh" || true
"$ROOT_DIR/scripts/stop.sh" || true
"$ROOT_DIR/scripts/start.sh"

base="$(server_base_url)"
for _ in $(seq 1 60); do
  if curl -fsS "$base/health/live" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

ready_timeout="${FULL_RESTART_READY_TIMEOUT:-1800}"
for _ in $(seq 1 "$ready_timeout"); do
  if curl -fsS "$base/health/ready?all=1" >/dev/null 2>&1; then
    "$ROOT_DIR/scripts/run_tunnel.sh"
    exit 0
  fi
  sleep 1
done

echo "No GPU worker became ready. Inspect log/server.log and log/worker-*.log." >&2
exit 1
