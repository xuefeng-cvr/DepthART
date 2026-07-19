#!/usr/bin/env python3
"""Build target-local engines and run the formal 36-case speed matrix."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCAN_ROOT = HERE / "selective_scan"
ENCODERS = ("S", "B", "L")
MODES = ("fp32", "amp", "trt_fp32", "trt_fp16")


def configurations():
    for encoder in ENCODERS:
        for resolution in (224, 448):
            yield "relative", encoder, resolution
    for encoder in ENCODERS:
        yield "metric", encoder, 448


def checkpoint(family, encoder, resolution, domain):
    if family == "relative":
        return ROOT / "checkpoints" / "relative" / f"depthart_relative_{encoder.lower()}_{resolution}.pth"
    return ROOT / "checkpoints" / "metric" / f"depthart_metric_{domain}_{encoder.lower()}_448.pth"


def run(command, log_path, environment):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND: " + " ".join(map(str, command)) + "\n")
        stream.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}); see {log_path}")
    return time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "export", "build", "benchmark", "summary"),
        default="all",
    )
    parser.add_argument("--metric-domain", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "deploy" / "runs" / "local",
        help="directory for target-local engines, logs, raw results, and summary.csv",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    artifact_root = args.output_root.resolve()
    onnx_root = HERE / "onnx"
    engine_root = artifact_root / "engines"
    result_root = artifact_root / "results"
    log_root = artifact_root / "logs"
    for path in (onnx_root, engine_root, result_root, log_root):
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join((str(SCAN_ROOT), environment.get("PYTHONPATH", "")))
    plugin = SCAN_ROOT / "results/plugins/libdepthart_selective_scan_trt.so"
    failures = []

    if args.stage in ("all", "export"):
        for family, encoder, resolution in configurations():
            stem = f"{family}_{encoder.lower()}_{resolution}"
            variants = (("default", False),) if family == "relative" else (("kv", True), ("fused", False))
            for variant, cache_kv in variants:
                output = onnx_root / f"{stem}_{variant}.onnx"
                case = f"{stem}_{variant}"
                if output.is_file() and not args.force:
                    print(f"[export skip] {case}", flush=True)
                    continue
                command = [
                    sys.executable,
                    str(HERE / "export_model.py"),
                    "--family", family,
                    "--encoder", encoder,
                    "--resolution", str(resolution),
                    "--checkpoint", str(checkpoint(family, encoder, resolution, args.metric_domain)),
                    "--metric-domain", args.metric_domain,
                    "--output", str(output),
                    "--device", "cuda",
                ]
                if cache_kv:
                    command.append("--cache-daa-kv")
                try:
                    elapsed = run(command, log_root / f"export_{case}.log", environment)
                    print(f"[export done {elapsed:.1f}s] {case}", flush=True)
                except Exception as error:
                    failures.append({"stage": "export", "case": case, "error": str(error)})
                    print(f"[export FAIL] {case}: {error}", flush=True)

    if args.stage in ("all", "build"):
        if not plugin.is_file():
            raise FileNotFoundError(f"missing TensorRT plugin; run install.sh first: {plugin}")
        for family, encoder, resolution in configurations():
            stem = f"{family}_{encoder.lower()}_{resolution}"
            for precision in ("tf32", "fp16"):
                variant = "default" if family == "relative" else ("kv" if precision == "tf32" else "fused")
                onnx_path = onnx_root / f"{stem}_{variant}.onnx"
                engine = engine_root / f"{stem}_{precision}.engine"
                case = f"{stem}_{precision}"
                if not onnx_path.is_file():
                    failures.append({"stage": "build", "case": case, "error": "missing ONNX"})
                    continue
                if engine.is_file() and not args.force:
                    print(f"[build skip] {case}", flush=True)
                    continue
                command = [
                    sys.executable,
                    str(HERE / "build_trt_engine.py"),
                    "--onnx", str(onnx_path),
                    "--plugin", str(plugin),
                    "--output", str(engine),
                    "--precision", precision,
                    "--workspace-gb", str(args.workspace_gb),
                    "--timing-cache", str(engine_root / f"timing_{precision}.cache"),
                ]
                if family == "relative" and encoder == "L" and resolution == 224 and precision == "fp16":
                    command.append("--output-head-fp32")
                try:
                    elapsed = run(command, log_root / f"build_{case}.log", environment)
                    print(f"[build done {elapsed:.1f}s] {case}", flush=True)
                except Exception as error:
                    failures.append({"stage": "build", "case": case, "error": str(error)})
                    print(f"[build FAIL] {case}: {error}", flush=True)

    if args.stage in ("all", "benchmark"):
        for family, encoder, resolution in configurations():
            stem = f"{family}_{encoder.lower()}_{resolution}"
            ckpt = checkpoint(family, encoder, resolution, args.metric_domain)
            for mode in MODES:
                case = f"{stem}_{mode}"
                output = result_root / f"{case}.json"
                if output.is_file() and not args.force:
                    print(f"[bench skip] {case}", flush=True)
                    continue
                command = [
                    sys.executable,
                    str(HERE / "benchmark.py"),
                    "--family", family,
                    "--encoder", encoder,
                    "--resolution", str(resolution),
                    "--mode", mode,
                    "--checkpoint", str(ckpt),
                    "--metric-domain", args.metric_domain,
                    "--samples", str(args.samples),
                    "--warmup", str(args.warmup),
                    "--plugin", str(plugin),
                    "--output", str(output),
                ]
                if mode.startswith("trt"):
                    precision = "tf32" if mode == "trt_fp32" else "fp16"
                    engine = engine_root / f"{stem}_{precision}.engine"
                    if not engine.is_file():
                        failures.append({"stage": "benchmark", "case": case, "error": "missing engine"})
                        continue
                    command.extend(("--engine", str(engine)))
                try:
                    elapsed = run(command, log_root / f"bench_{case}.log", environment)
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    if not payload.get("numerical_validation_passed"):
                        raise RuntimeError("numerical validation failed")
                    print(f"[bench done {elapsed:.1f}s] {case}", flush=True)
                except Exception as error:
                    failures.append({"stage": "benchmark", "case": case, "error": str(error)})
                    print(f"[bench FAIL] {case}: {error}", flush=True)

    report = {"failures": failures, "failure_count": len(failures)}
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "failures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)

    if args.stage in ("all", "benchmark", "summary"):
        command = [
            sys.executable,
            str(HERE / "summarize_results.py"),
            "--results", str(result_root),
            "--output", str(artifact_root / "summary.csv"),
            "--samples", str(args.samples),
            "--warmup", str(args.warmup),
        ]
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
