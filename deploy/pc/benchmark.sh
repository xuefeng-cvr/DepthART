#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${DEPTHART_OUTPUT_ROOT:-$HERE/runs/local}"
python3 "$HERE/../shared/run_speed_matrix.py" \
  --output-root "$OUTPUT_ROOT" \
  "$@"
