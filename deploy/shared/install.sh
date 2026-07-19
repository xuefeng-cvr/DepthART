#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-native}"
MODE="${2:-full}"
cd "$ROOT"

if [[ "$MODE" != "full" && "$MODE" != "pytorch-only" ]]; then
    echo "usage: $0 {native|orin-nx|xavier-nx|jetson-nano} {full|pytorch-only}" >&2
    exit 2
fi

python3 -m pip install -r requirements.txt
MAX_JOBS="${MAX_JOBS:-4}" bash selective_scan/scripts/build_for_device.sh "$TARGET"
if [[ "$MODE" == "full" ]]; then
    MAX_JOBS="${MAX_JOBS:-4}" bash selective_scan/scripts/build_trt_plugin.sh "$TARGET"
fi

DEPTHART_INSTALL_MODE="$MODE" python3 - <<'PY'
import ctypes
import os
from pathlib import Path

import torch
from depthart_selective_scan import selective_scan

plugin = Path("selective_scan/results/plugins/libdepthart_selective_scan_trt.so").resolve()
if os.environ["DEPTHART_INSTALL_MODE"] == "full":
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", ".".join(map(str, torch.cuda.get_device_capability(0))))
print("PyTorch selective scan: OK")
print("TensorRT plugin:", plugin if plugin.is_file() else "not built (PyTorch-only mode)")
PY
