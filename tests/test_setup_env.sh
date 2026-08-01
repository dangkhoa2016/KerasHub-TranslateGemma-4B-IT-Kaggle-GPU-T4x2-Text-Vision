#!/usr/bin/env bash
# Regression test: scripts/setup.sh must create .env before loading it so a
# fresh checkout works in a single invocation. Runs against copies of the
# scripts inside a sandbox and never mutates the real working tree.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

mkdir -p "$SANDBOX/scripts"
cp "$ROOT/scripts/setup.sh" "$SANDBOX/scripts/setup.sh"
cp "$ROOT/scripts/_common.sh" "$SANDBOX/scripts/_common.sh"
cp "$ROOT/.env.example" "$SANDBOX/.env.example"

CACHE_DIR="$SANDBOX/.cache/jax"
# The sentinel lets the test prove the freshly-created .env is loaded during
# the same invocation: only then does setup.sh see JAX_COMPILATION_CACHE_DIR
# and print the cache message before failing on missing Kaggle dependencies.
cat >> "$SANDBOX/.env.example" <<EOF
JAX_COMPILATION_CACHE_DIR=$CACHE_DIR
EOF

output="$SANDBOX/setup.out"
(
  cd "$SANDBOX"
  bash scripts/setup.sh >"$output" 2>&1 || true
)

if [[ ! -f "$SANDBOX/.env" ]]; then
  echo "FAIL: setup.sh did not create .env from .env.example" >&2
  exit 1
fi

if ! grep -q "^JAX_COMPILATION_CACHE_DIR=$CACHE_DIR$" "$SANDBOX/.env"; then
  echo "FAIL: created .env does not contain the sentinel variable" >&2
  exit 1
fi

if ! grep -q "JAX compilation cache: $CACHE_DIR" "$output"; then
  echo "FAIL: setup.sh did not load the freshly-created .env in the same invocation" >&2
  echo "--- setup.sh output ---" >&2
  cat "$output" >&2
  exit 1
fi

echo "PASS: setup.sh creates .env and loads it in the same invocation"
