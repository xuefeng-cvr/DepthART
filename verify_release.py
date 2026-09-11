#!/usr/bin/env python3
"""Validate the portable release inventory without loading checkpoints on GPU."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch


ROOT = Path(__file__).resolve().parent


def verify_checksums(path):
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        target = ROOT / relative_path.lstrip("*")
        if not target.is_file():
            failures.append(f"missing {relative_path}")
            continue
        hasher = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != expected:
            failures.append(f"checksum mismatch {relative_path}")
    if failures:
        raise RuntimeError(f"{path.name}: {failures}")


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

    verify_checksums(ROOT / "CHECKSUMS.sha256")
    verify_checksums(ROOT / "RELEASE_MANIFEST.sha256")

    expected_checkpoints = {
        *(f"depthart_relative_{scale}_{resolution}.pth" for scale in "sbl" for resolution in (224, 448)),
        *(f"depthart_relative_mnv4{scale}_{resolution}.pth" for scale in ("s", "m") for resolution in (224, 448)),
        *(f"depthart_relative_mnv4m_slim_{resolution}.pth" for resolution in (224, 448)),
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
    for folder in (ROOT / "checkpoints/relative", ROOT / "checkpoints/metric"):
        for path in folder.iterdir():
            if not path.is_file():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if set(payload) != {"model", "validation_metrics"}:
                raise RuntimeError(
                    f"{path}: expected only model and validation_metrics, "
                    f"found {sorted(payload)}"
                )

    sys.path.insert(0, str(ROOT / "relative"))
    from models import load_relative_model
    from utils.tiled_kitti import hann_blend, joint_affine_align

    reference = np.arange(96, dtype=np.float32).reshape(8, 12) / 32.0
    starts = [0, 4]
    tiles = np.stack((2.0 * reference[:, :8] + 3.0, reference[:, 4:]))
    aligned = joint_affine_align(
        tiles, starts, sample_stride=1, minimum_overlap=1, trim_quantile=1.0
    )
    if not np.allclose(aligned[0, :, 4:], aligned[1, :, :4], atol=1e-5):
        raise RuntimeError("KITTI tile overlap affine alignment failed")
    if hann_blend(aligned, starts, reference.shape[1]).shape != reference.shape:
        raise RuntimeError("KITTI Hann blending returned an unexpected shape")

    mobilenet_variants = {
        "mnv4s": "MNV4-S",
        "mnv4m": "MNV4-M",
        "mnv4m_slim": "MNV4-M-SLIM-SPF",
    }
    for checkpoint_stem, encoder in mobilenet_variants.items():
        for resolution in (224, 448):
            checkpoint = (
                ROOT
                / "checkpoints/relative"
                / f"depthart_relative_{checkpoint_stem}_{resolution}.pth"
            )
            load_relative_model(encoder, checkpoint, "cpu")

    a6000 = ROOT / "deploy/pc/results/a6000"
    csv_rows(a6000 / "summary.csv", 36)
    csv_rows(a6000 / "depth_eval/depth_metrics.csv", 18)
    csv_rows(a6000 / "depth_eval/backend_depth_metrics.csv", 54)
    csv_rows(a6000 / "mobilenetv4_relative_accuracy.csv", 12)
    csv_rows(a6000 / "mobilenetv4_relative_speed.csv", 3)
    csv_rows(a6000 / "complete_depth_accuracy.csv", 90)
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

    expected_mobile_onnx = {
        f"relative_{variant}_{resolution}_default.onnx"
        for variant in ("mnv4s", "mnv4m", "mnv4m_slim_spf")
        for resolution in (224, 448)
    }
    onnx_paths = sorted((ROOT / "deploy/shared/onnx").glob("*.onnx"))
    if len(onnx_paths) != 18:
        raise RuntimeError(f"expected 18 ONNX files, found {len(onnx_paths)}")
    actual_mobile_onnx = {
        path.name for path in onnx_paths if path.name.startswith("relative_mnv4")
    }
    if actual_mobile_onnx != expected_mobile_onnx:
        raise RuntimeError(
            "MobileNetV4 ONNX inventory mismatch: "
            f"missing={sorted(expected_mobile_onnx - actual_mobile_onnx)}, "
            f"unexpected={sorted(actual_mobile_onnx - expected_mobile_onnx)}"
        )
    for path in onnx_paths:
        graph = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(graph)
        scans = [
            node for node in graph.graph.node
            if node.domain == "com.depthart" and node.op_type == "SelectiveScan"
        ]
        if path.name in expected_mobile_onnx and scans:
            raise RuntimeError(f"{path.name}: unexpected custom SelectiveScan nodes")
        if path.name not in expected_mobile_onnx and not scans:
            raise RuntimeError(f"{path.name}: no custom SelectiveScan nodes")

    forbidden_suffixes = {
        ".engine", ".cache", ".timing", ".so", ".log", ".pid", ".pyc"
    }
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
        if ".git" not in path.parts and path.is_file() and (
            path.suffix in forbidden_suffixes
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or "3090" in path.name.lower()
            or path.name.startswith("wget-log")
        ) and not is_generated_build_artifact(path)
    ]
    if forbidden:
        raise RuntimeError(f"non-portable or obsolete files remain: {forbidden}")
    print("release validation: OK")
    print("checksums: model artifacts and release manifest verified")
    print("checkpoints: 18 safe inference-only files in two flat task directories")
    print("MobileNetV4: 6 checkpoints loaded strictly on CPU")
    print("KITTI tiled inference: overlap affine alignment and Hann blending checked")
    print("A6000: 36 speed rows, 18 baseline accuracy rows, 54 backend accuracy rows")
    print("Comparison: 90 complete quantitative rows across five datasets")
    print("portable ONNX: 18 checked graphs (6 standard MobileNetV4 graphs)")
    if args.allow_build_artifacts:
        print("prepackaged engines/cache/logs/RTX3090 artifacts: 0 (local build libraries allowed)")
    else:
        print("engines/cache/shared libraries/logs/RTX3090 artifacts: 0")


if __name__ == "__main__":
    main()
