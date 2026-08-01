#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

# Load environment only after .env is guaranteed to exist so a fresh checkout
# works in a single invocation.
load_env
export KERAS_BACKEND=jax

if [[ -n "${JAX_COMPILATION_CACHE_DIR:-}" ]]; then
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
  chmod 700 "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || true
  echo "JAX compilation cache: $JAX_COMPILATION_CACHE_DIR"
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
print("Python:", sys.version.split()[0])
PY

check_python_environment() {
  python3 - <<'PY'
import flask
import keras
import keras_hub
import jax
print("Flask: installed")
print("Keras:", getattr(keras, "__version__", "unknown"))
print("KerasHub:", getattr(keras_hub, "__version__", "unknown"))
print("JAX:", getattr(jax, "__version__", "unknown"))
print("JAX devices:", jax.devices())
PY
}

if ! check_python_environment; then
  if [[ "${INSTALL_PYTHON_DEPS:-0}" == "1" ]]; then
    python3 -m pip install -U -r requirements.txt -c constraints-kaggle-tested.txt
    if ! check_python_environment; then
      cat >&2 <<'MSG'
Flask/Keras/KerasHub were installed, but JAX is still missing or unusable.
Use the CUDA-enabled JAX/JAXLIB pair supplied by Kaggle. Do not install a
normal CPU-only JAX wheel over the Kaggle environment.
MSG
      exit 1
    fi
  else
    cat >&2 <<'MSG'
Python dependencies are missing or unusable.
Re-run with INSTALL_PYTHON_DEPS=1 to install Flask/Keras/KerasHub.
Do not replace Kaggle's CUDA-enabled JAX wheel with a generic CPU wheel.
MSG
    exit 1
  fi
fi

MODEL_PATH="${MODEL_PATH:-/kaggle/input/models/keras/translategemma/keras/translategemma_4b_it/1}"
model_complete() {
  local base="$1"
  [[ -f "$base/config.json" \
    && -f "$base/preprocessor.json" \
    && -f "$base/model.weights.h5" \
    && -f "$base/assets/tokenizer/vocabulary.spm" ]]
}
if ! model_complete "$MODEL_PATH" && [[ "${MODEL_AUTO_DISCOVER:-true}" != "false" ]]; then
  model_base="/kaggle/input/models/keras/translategemma/keras/translategemma_4b_it"
  discovered=""
  if [[ -d "$model_base" ]]; then
    while IFS= read -r candidate; do
      if model_complete "$candidate"; then discovered="$candidate"; break; fi
    done < <(find "$model_base" -mindepth 1 -maxdepth 1 -type d -print | sort -V -r)
  fi
  if [[ -n "$discovered" ]]; then
    echo "MODEL_PATH '$MODEL_PATH' is unavailable; auto-discovered: $discovered"
    export MODEL_PATH="$discovered"
  fi
fi
if ! model_complete "$MODEL_PATH"; then
  echo "TranslateGemma checkpoint is incomplete at: $MODEL_PATH" >&2
  echo "Mount the model, update MODEL_PATH, or leave MODEL_AUTO_DISCOVER=true." >&2
  exit 1
fi

CLOUDFLARED_BIN="$ROOT_DIR/bin/cloudflared"
arch="$(uname -m)"
version="${CLOUDFLARED_VERSION:-2026.7.3}"
case "$arch" in
  x86_64|amd64)
    asset="cloudflared-linux-amd64"
    expected_sha="${CLOUDFLARED_SHA256:-9d71c677db00134c1bd4144b7783486b654ad281b1ea62b4972098d19f770f17}"
    ;;
  aarch64|arm64)
    asset="cloudflared-linux-arm64"
    expected_sha="${CLOUDFLARED_SHA256:-65259e652a7bea08bf5df603233ab22b8bf3116af8df9f9206209af6a1b955c0}"
    ;;
  *)
    echo "Unsupported architecture for automatic cloudflared install: $arch" >&2
    exit 1
    ;;
esac

cloudflared_sha_ok() {
  local path="$1" actual
  [[ -f "$path" ]] || return 1
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected_sha" ]]
}

if [[ -f "$CLOUDFLARED_BIN" ]] && cloudflared_sha_ok "$CLOUDFLARED_BIN"; then
  chmod 755 "$CLOUDFLARED_BIN"
  echo "Using bundled cloudflared $version: $CLOUDFLARED_BIN"
  echo "cloudflared SHA256: OK"
elif [[ -f "$CLOUDFLARED_BIN" ]]; then
  echo "Bundled cloudflared failed SHA256 verification; it will not be executed." >&2
  if [[ "${INSTALL_CLOUDFLARED:-1}" != "1" ]]; then
    exit 1
  fi
  rm -f "$CLOUDFLARED_BIN"
elif command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared from PATH: $(command -v cloudflared)"
fi

if [[ ! -f "$CLOUDFLARED_BIN" ]] && ! command -v cloudflared >/dev/null 2>&1; then
  if [[ "${INSTALL_CLOUDFLARED:-1}" == "1" ]]; then
    mkdir -p "$ROOT_DIR/bin"
    tmp="$ROOT_DIR/bin/cloudflared.tmp.$$"
    echo "Downloading pinned cloudflared $version to $CLOUDFLARED_BIN..."
    curl -fL --retry 3 \
      "https://github.com/cloudflare/cloudflared/releases/download/$version/$asset" \
      -o "$tmp"
    actual_sha="$(sha256sum "$tmp" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      rm -f "$tmp"
      echo "cloudflared SHA256 mismatch; refusing to execute downloaded binary." >&2
      exit 1
    fi
    chmod 755 "$tmp"
    mv -f "$tmp" "$CLOUDFLARED_BIN"
    echo "Downloaded and verified cloudflared $version."
  else
    echo "cloudflared not installed; tunnel scripts will be unavailable." >&2
  fi
fi

python3 scripts/capture_versions.py
python3 -m py_compile src/server.py src/translategemma/*.py src/translategemma/*/*.py
echo "Setup checks passed."
