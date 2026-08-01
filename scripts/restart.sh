#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env
"$ROOT_DIR/scripts/stop.sh"
"$ROOT_DIR/scripts/start.sh"
