#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

device="${1:-native}"
case "${device}" in
  native)
    unset TORCH_CUDA_ARCH_LIST
    ;;
  orin-nx)
    export TORCH_CUDA_ARCH_LIST="8.7"
    ;;
  xavier-nx)
    export TORCH_CUDA_ARCH_LIST="7.2"
    ;;
  jetson-nano)
    export TORCH_CUDA_ARCH_LIST="5.3"
    export DEPTHART_CXX_STANDARD="c++14"
    ;;
  *)
    echo "usage: $0 {native|orin-nx|xavier-nx|jetson-nano}" >&2
    exit 2
    ;;
esac

echo "device=${device}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-native}"
echo "DEPTHART_CXX_STANDARD=${DEPTHART_CXX_STANDARD:-auto}"
python -m pip install -v . --no-build-isolation --force-reinstall --no-deps
