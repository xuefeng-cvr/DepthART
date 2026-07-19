#!/usr/bin/env python3
"""Build the formal 36-row Orin NX speed summary from raw benchmark JSON."""

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE.parent / "runs/local/results")
    parser.add_argument("--output", type=Path, default=HERE.parent / "runs/local/summary.csv")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        numerical = payload.get("numerical_vs_original_pytorch_fp32") or {}
        system = payload.get("system") or {}
        gpu = system.get("gpu_after") or system.get("gpu_before") or {}
        rows.append({
            "family": payload["family"],
            "encoder": payload["encoder"],
            "resolution": payload["resolution"],
            "mode": payload["mode"],
            "tf32_allowed": payload["tf32"],
            "model_mean_ms": payload["model_only"]["mean_ms"],
            "model_p50_ms": payload["model_only"]["p50_ms"],
            "model_p95_ms": payload["model_only"]["p95_ms"],
            "model_fps": payload["model_only"]["fps_from_mean"],
            "e2e_mean_ms": payload["end_to_end"]["mean_ms"],
            "e2e_p50_ms": payload["end_to_end"]["p50_ms"],
            "e2e_p95_ms": payload["end_to_end"]["p95_ms"],
            "e2e_fps": payload["end_to_end"]["fps_from_mean"],
            "peak_mib": payload["peak_torch_allocated_mib"],
            "samples": payload["samples"],
            "warmup": payload["warmup"],
            "trt_validation_passed": (
                payload["numerical_validation_passed"] if payload["mode"].startswith("trt") else ""
            ),
            "finite": numerical.get("finite", ""),
            "shape_match": numerical.get("shape_match", ""),
            "relative_mae": numerical.get("relative_mae", ""),
            "affine_aligned_relative_mae": numerical.get("affine_aligned_relative_mae", ""),
            "parameters_unchanged": payload.get("parameters_unchanged", ""),
            "parameter_sha256": payload.get("parameter_sha256", ""),
            "checkpoint": payload["checkpoint"],
            "engine": payload.get("engine") or "",
            "plugin": payload.get("plugin") or "",
            "backend": payload.get("backend", ""),
            "gpu": gpu.get("name", ""),
            "driver": gpu.get("driver", ""),
            "torch": system.get("torch", ""),
            "torch_cuda_build": system.get("torch_cuda_build", ""),
            "cudnn": system.get("cudnn", ""),
            "tensorrt": system.get("tensorrt", ""),
        })

    if len(rows) != 36:
        raise RuntimeError(f"expected 36 result rows, found {len(rows)}")
    keys = {(row["family"], row["encoder"], int(row["resolution"]), row["mode"]) for row in rows}
    if len(keys) != 36:
        raise RuntimeError("duplicate benchmark combinations found")
    if not all(int(row["samples"]) == args.samples and int(row["warmup"]) == args.warmup for row in rows):
        raise RuntimeError(f"formal summary requires samples={args.samples}, warmup={args.warmup}")
    if not all(row["finite"] is True and row["shape_match"] is True for row in rows):
        raise RuntimeError("one or more numerical validations failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
