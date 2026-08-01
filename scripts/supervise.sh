#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

if [[ "${_SUPERVISE_BG:-0}" != "1" ]]; then
  mkdir -p "$ROOT_DIR/log"
  nohup env _SUPERVISE_BG=1 bash "$(realpath "$0")" "$@" \
    > "$ROOT_DIR/log/supervise.stdout.log" 2>&1 &
  echo "supervise.sh started in background (PID $!). Log: log/supervise.stdout.log"
  exit 0
fi

max_restarts="${MAX_SERVER_RESTARTS:-10}"
restart_count=0

env _FULL_RESTART_BG=1 bash "$ROOT_DIR/scripts/full_restart.sh"
while (( restart_count < max_restarts )); do
  pid="$(cat "$ROOT_DIR/state/server.pid" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    echo "Missing server PID file." >&2
  elif kill -0 "$pid" 2>/dev/null; then
    sleep 5
    continue
  fi

  restart_count=$((restart_count + 1))
  echo "API coordinator stopped; restart $restart_count/$max_restarts" >&2
  "$ROOT_DIR/scripts/start.sh"

  for _ in $(seq 1 "${SUPERVISOR_READY_TIMEOUT:-1800}"); do
    if curl -fsS "$(server_base_url)/health/ready?all=1" >/dev/null 2>&1; then
      break
    fi
    pid="$(cat "$ROOT_DIR/state/server.pid" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
done

echo "Maximum server restart count reached." >&2
exit 1
