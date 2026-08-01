#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
pid_file="$ROOT_DIR/state/tunnel.pid"
if pid="$(read_managed_pid "$pid_file" "cloudflared" 2>/dev/null)"; then
  kill -TERM "$pid" 2>/dev/null || true
  rm -f "$pid_file"
else
  rm -f "$pid_file"
fi
rm -f "$ROOT_DIR/data/tunnel_url.txt"
echo "Tunnel stopped."
