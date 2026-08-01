#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

if command -v cloudflared >/dev/null 2>&1; then
  cloudflared_bin="$(command -v cloudflared)"
elif [[ -x "$ROOT_DIR/bin/cloudflared" ]]; then
  cloudflared_bin="$ROOT_DIR/bin/cloudflared"
else
  echo "cloudflared is missing. Run scripts/setup.sh first." >&2
  exit 1
fi

pid_file="$ROOT_DIR/state/tunnel.pid"
if old_pid="$(read_managed_pid "$pid_file" "cloudflared" 2>/dev/null)"; then
  kill -TERM "$old_pid" 2>/dev/null || true
  sleep 1
fi

: > log/tunnel.log
nohup "$cloudflared_bin" tunnel --no-autoupdate \
  --url "http://127.0.0.1:$PORT" > log/tunnel.log 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$pid_file"

url=""
for _ in $(seq 1 60); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "cloudflared exited; inspect log/tunnel.log" >&2
    exit 1
  fi
  url="$(grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' log/tunnel.log | tail -n1 || true)"
  if [[ -n "$url" ]]; then
    printf '%s\n' "$url" > data/tunnel_url.txt
    echo "Tunnel URL: $url"
    exit 0
  fi
  sleep 1
done

echo "Tunnel started but no public URL was detected; inspect log/tunnel.log" >&2
exit 1
