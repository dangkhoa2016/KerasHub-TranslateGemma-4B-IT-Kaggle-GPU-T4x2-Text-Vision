#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p data log state bin

load_env() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
  fi
  export PORT="${PORT:-7860}"
  export HOST="${HOST:-0.0.0.0}"
}

api_key() {
  if [[ -n "${API_KEY:-}" ]]; then
    printf '%s' "$API_KEY"
  elif [[ -s "$ROOT_DIR/data/api_key.txt" ]]; then
    tr -d '\r\n' < "$ROOT_DIR/data/api_key.txt"
  fi
}

restart_secret() {
  if [[ -n "${RESTART_SECRET:-}" ]]; then
    printf '%s' "$RESTART_SECRET"
  elif [[ -s "$ROOT_DIR/data/restart_secret.txt" ]]; then
    tr -d '\r\n' < "$ROOT_DIR/data/restart_secret.txt"
  fi
}

server_base_url() {
  printf 'http://127.0.0.1:%s' "$PORT"
}


pid_matches_cmdline() {
  local pid="${1:-}" marker="${2:-}" cmdline
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *"$marker"* ]]
}

read_managed_pid() {
  local pid_file="${1:-}" marker="${2:-}" pid
  [[ -s "$pid_file" ]] || return 1
  read -r pid < "$pid_file" || return 1
  if pid_matches_cmdline "$pid" "$marker"; then
    printf '%s' "$pid"
    return 0
  fi
  rm -f "$pid_file"
  return 1
}


require_vision_ready_pool() {
  local base_url="${1:-$(server_base_url)}"
  local key="${2:-$(api_key)}"
  local health_json
  local headers=()

  if [[ "${VISION_ENABLED:-false}" != "true" ]]; then
    echo "VISION_ENABLED must be true for the T4x2 vision test workflow." >&2
    return 1
  fi
  python3 - "${JAX_MEM_FRACTION:-0}" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("JAX_MEM_FRACTION must be numeric")
if value < 0.97:
    raise SystemExit("Vision mode requires JAX_MEM_FRACTION >= 0.97")
PY

  if [[ "${API_AUTH_REQUIRED:-true}" != "false" ]]; then
    if [[ -z "$key" ]]; then
      echo "API key is unavailable; cannot inspect detailed worker health." >&2
      return 1
    fi
    headers=(-H "Authorization: Bearer $key")
  fi

  health_json="$(curl -fsS "${headers[@]}" "$base_url/health/ready?all=1&details=1")" || {
    echo "The complete worker pool is not ready." >&2
    return 1
  }

  python3 -c '
import json, sys
health = json.load(sys.stdin)
workers = health.get("workers") or []
expected = int(health.get("expected_workers") or 0)
ready = int(health.get("ready_workers") or 0)
if expected < 1 or ready != expected:
    raise SystemExit(f"Expected the full worker pool, got {ready}/{expected} ready.")
if not workers:
    raise SystemExit("Detailed health response contains no worker metadata.")
bad = []
for worker in workers:
    state = worker.get("state")
    metadata = worker.get("metadata") or {}
    if state not in {"ready", "busy"} or metadata.get("vision_enabled") is not True:
        bad.append(str(worker.get("worker_id", "unknown")))
if bad:
    raise SystemExit("Running worker(s) are not ready multimodal/vision workers: " + ", ".join(bad))
print(f"Verified multimodal vision-enabled worker pool: {ready}/{expected} ready.")
' <<<"$health_json"
}
