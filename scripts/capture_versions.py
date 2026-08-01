#!/usr/bin/env python3
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGE_DISTRIBUTIONS = {
    "flask": "flask",
    "keras": "keras",
    "keras_hub": "keras-hub",
    "jax": "jax",
    "jaxlib": "jaxlib",
}

packages = {}
for name, distribution in PACKAGE_DISTRIBUTIONS.items():
    try:
        packages[name] = version(distribution)
    except PackageNotFoundError:
        packages[name] = "unavailable"
    except Exception as exc:
        packages[name] = f"unavailable: {exc!r}"

# Device discovery still imports JAX intentionally; unlike importing all five
# packages above, this is needed to prove the CUDA-enabled runtime is usable.
try:
    import jax

    devices = [str(device) for device in jax.devices()]
except Exception as exc:
    devices = [f"unavailable: {exc!r}"]

try:
    gpu_text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=10,
    ).strip()
    gpus = gpu_text.splitlines()
except Exception:
    gpus = []

payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": packages,
    "jax_devices": devices,
    "gpus": gpus,
}
path = Path(__file__).resolve().parents[1] / "data" / "environment.json"
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {path}")
