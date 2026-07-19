#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-native}"
case "$TARGET" in
    native)
        : "${TORCH_CUDA_ARCH_LIST:=$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))')}"
        ;;
    orin-nx)
        : "${TORCH_CUDA_ARCH_LIST:=8.7}"
        ;;
    xavier-nx)
        : "${TORCH_CUDA_ARCH_LIST:=7.2}"
        ;;
    jetson-nano)
        : "${TORCH_CUDA_ARCH_LIST:=5.3}"
        : "${DEPTHART_CXX_STANDARD:=c++14}"
        export DEPTHART_CXX_STANDARD
        ;;
    *)
        echo "unknown target: $TARGET" >&2
        exit 2
        ;;
esac
: "${MAX_JOBS:=4}"
export TORCH_CUDA_ARCH_LIST MAX_JOBS

python setup_trt_plugin.py build_ext --inplace

mkdir -p results/plugins
PLUGIN="$(find . -maxdepth 1 -type f -name 'depthart_selective_scan_trt_plugin*.so' -print -quit)"
if [[ -z "$PLUGIN" ]]; then
    echo "TensorRT plugin build completed but no shared library was found" >&2
    exit 1
fi
cp "$PLUGIN" results/plugins/libdepthart_selective_scan_trt.so
echo "plugin: $ROOT/results/plugins/libdepthart_selective_scan_trt.so"
