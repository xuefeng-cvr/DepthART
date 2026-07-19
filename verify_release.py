#!/usr/bin/env python3
"""Validate the portable release inventory without loading checkpoints on GPU."""

import argparse
import csv
import json
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parent


def csv_rows(path, expected):
    with path.open(newline="", encoding="utf-8") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} rows, found {count}")


def file_count(path, pattern, expected):
    count = sum(1 for _ in path.glob(pattern))
    if count != expected:
        raise RuntimeError(f"{path}/{pattern}: expected {expected}, found {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-build-artifacts",
        action="store_true",
        help="ignore local Selective Scan libraries created by the setup scripts",
    )
    args = parser.parse_args()

    expected_checkpoints = {
        *(f"depthart_relative_{scale}_{resolution}.pth" for scale in "sbl" for resolution in (224, 448)),
        *(f"depthart_metric_{domain}_{scale}_448.pth" for domain in ("indoor", "outdoor") for scale in "sbl"),
    }
    actual_checkpoints = {
        path.name for folder in (ROOT / "checkpoints/relative", ROOT / "checkpoints/metric")
        for path in folder.iterdir() if path.is_file()
    }
    if actual_checkpoints != expected_checkpoints:
        raise RuntimeError(
            f"checkpoint inventory mismatch: missing={sorted(expected_checkpoints - actual_checkpoints)}, "
            f"unexpected={sorted(actual_checkpoints - expected_checkpoints)}"
        )
    checkpoint_subdirectories = [
        path for folder in (ROOT / "checkpoints/relative", ROOT / "checkpoints/metric")
        for path in folder.iterdir() if path.is_dir()
    ]
    if checkpoint_subdirectories:
        raise RuntimeError(f"checkpoint directories must be flat: {checkpoint_subdirectories}")

    a6000 = ROOT / "deploy/pc/results/a6000"
    csv_rows(a6000 / "summary.csv", 36)
    csv_rows(a6000 / "depth_eval/depth_metrics.csv", 18)
    csv_rows(a6000 / "depth_eval/backend_depth_metrics.csv", 54)
    file_count(a6000 / "speed_results", "*.json", 36)
    file_count(a6000 / "depth_eval/metric_results", "*.json", 6)
    file_count(a6000 / "depth_eval/backend_results", "*.json", 54)

    failures = []
    for path in (a6000 / "depth_eval/backend_results").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("metrics"):
            failures.append(path.name)
    if failures:
        raise RuntimeError(f"backend results without metrics: {failures}")

    onnx_paths = sorted((ROOT / "deploy/shared/onnx").glob("*.onnx"))
    if len(onnx_paths) != 12:
        raise RuntimeError(f"expected 12 ONNX files, found {len(onnx_paths)}")
    for path in onnx_paths:
        graph = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(graph)
        scans = [
            node for node in graph.graph.node
            if node.domain == "com.depthart" and node.op_type == "SelectiveScan"
        ]
        if not scans:
            raise RuntimeError(f"{path.name}: no custom SelectiveScan nodes")

    forbidden_suffixes = {".engine", ".cache", ".timing", ".so", ".log", ".pid"}
    scan_root = ROOT / "deploy/shared/selective_scan"
    generated_roots = (scan_root / "build", scan_root / "results")

    def is_generated_build_artifact(path):
        if not args.allow_build_artifacts:
            return False
        if path.parent == scan_root and path.name.startswith("depthart_selective_scan_cuda"):
            return True
        return any(path.is_relative_to(folder) for folder in generated_roots)

    forbidden = [
        path for path in ROOT.rglob("*")
        if path.is_file() and (
            path.suffix in forbidden_suffixes
            or "3090" in path.name.lower()
            or path.name.startswith("wget-log")
        ) and not is_generated_build_artifact(path)
    ]
    if forbidden:
        raise RuntimeError(f"non-portable or obsolete files remain: {forbidden}")
    print("release validation: OK")
    print("checkpoints: 12 files in two flat task directories")
    print("A6000: 36 speed rows, 18 baseline accuracy rows, 54 backend accuracy rows")
    print("portable ONNX: 12 checked graphs")
    if args.allow_build_artifacts:
        print("prepackaged engines/cache/logs/RTX3090 artifacts: 0 (local build libraries allowed)")
    else:
        print("engines/cache/shared libraries/logs/RTX3090 artifacts: 0")


if __name__ == "__main__":
    main()
