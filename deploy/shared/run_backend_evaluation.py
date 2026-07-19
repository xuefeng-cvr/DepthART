#!/usr/bin/env python3
"""Export, build, evaluate, and summarize the DepthART backend matrix."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCAN_ROOT = Path(__file__).resolve().parent / "selective_scan"
OUTPUT_ROOT = ROOT / "deploy" / "runs" / "backend_evaluation"
PLUGIN = SCAN_ROOT / "results" / "plugins" / "libdepthart_selective_scan_trt.so"
BUILDER = HERE / "build_trt_engine.py"


def specs():
    relative_shapes = {
        (224, "NYUD"): (224, 288),
        (224, "KITTI"): (224, 736),
        (448, "NYUD"): (448, 608),
        (448, "KITTI"): (448, 1472),
    }
    for encoder in ("S", "B", "L"):
        for resolution in (224, 448):
            for dataset in ("NYUD", "KITTI"):
                height, width = relative_shapes[(resolution, dataset)]
                yield {
                    "family": "relative",
                    "encoder": encoder,
                    "resolution": resolution,
                    "dataset": dataset,
                    "height": height,
                    "width": width,
                    "domain": "relative",
                    "checkpoint": (
                        ROOT / "checkpoints" / "relative"
                        / f"depthart_relative_{encoder.lower()}_{resolution}.pth"
                    ),
                }
    for encoder in ("S", "B", "L"):
        for dataset, domain, shape in (
            ("NYUD", "indoor", (480, 640)),
            ("KITTI", "outdoor", (448, 1472)),
        ):
            yield {
                "family": "metric",
                "encoder": encoder,
                "resolution": 448,
                "dataset": dataset,
                "height": shape[0],
                "width": shape[1],
                "domain": domain,
                "checkpoint": (
                    ROOT / "checkpoints" / "metric"
                    / f"depthart_metric_{domain}_{encoder.lower()}_448.pth"
                ),
            }


def name(spec):
    return (
        f"{spec['family']}_{spec['encoder'].lower()}_{spec['resolution']}_"
        f"{spec['dataset'].lower()}_{spec['height']}x{spec['width']}"
    )


def run(command, cwd, log, label):
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    elapsed = time.monotonic() - started
    if result.returncode:
        tail = " | ".join(result.stdout.splitlines()[-8:])
        raise RuntimeError(f"{label} failed after {elapsed:.1f}s: {tail}")
    print(f"[{label} done {elapsed:.1f}s]", flush=True)


def export_all(args, failures):
    for spec in specs():
        item = name(spec)
        onnx_path = OUTPUT_ROOT / "onnx" / f"{item}.onnx"
        if onnx_path.is_file() and not args.force:
            print(f"[export skip] {item}", flush=True)
            continue
        command = [
            sys.executable, str(HERE / "export_backend_model.py"),
            "--family", spec["family"],
            "--encoder", spec["encoder"],
            "--height", str(spec["height"]),
            "--width", str(spec["width"]),
            "--checkpoint", str(spec["checkpoint"]),
            "--output", str(onnx_path),
        ]
        if spec["family"] == "metric":
            command += ["--metric-domain", spec["domain"]]
            if spec["dataset"] == "KITTI":
                command += ["--attention-chunk-size", "1024"]
        try:
            print(f"[export] {item}", flush=True)
            run(command, ROOT, OUTPUT_ROOT / "logs" / f"export_{item}.log", f"export {item}")
        except Exception as error:
            failures.append({"stage": "export", "item": item, "reason": str(error)})
            print(f"[export failed] {item}: {error}", flush=True)


def build_all(args, failures):
    for spec in specs():
        item = name(spec)
        onnx_path = OUTPUT_ROOT / "onnx" / f"{item}.onnx"
        if not onnx_path.is_file():
            failures.append({"stage": "build", "item": item, "reason": "missing ONNX"})
            continue
        for suffix, precision in (("trt_fp32", "tf32"), ("trt_fp16", "fp16")):
            engine = OUTPUT_ROOT / "engines" / f"{item}_{suffix}.engine"
            if engine.is_file() and not args.force:
                print(f"[build skip] {item}_{suffix}", flush=True)
                continue
            command = [
                sys.executable, str(BUILDER),
                "--onnx", str(onnx_path),
                "--plugin", str(PLUGIN),
                "--output", str(engine),
                "--precision", precision,
                "--workspace-gb", "8.0",
                "--timing-cache", str(OUTPUT_ROOT / "engines" / f"timing_{precision}.cache"),
            ]
            if (
                spec["family"] == "relative"
                and spec["encoder"] == "L"
                and spec["resolution"] == 224
                and precision == "fp16"
            ):
                command.append("--output-head-fp32")
            try:
                print(f"[build] {item}_{suffix}", flush=True)
                run(
                    command, ROOT,
                    OUTPUT_ROOT / "logs" / f"build_{item}_{suffix}.log",
                    f"build {item}_{suffix}",
                )
            except Exception as error:
                failures.append({"stage": "build", "item": f"{item}_{suffix}", "reason": str(error)})
                print(f"[build failed] {item}_{suffix}: {error}", flush=True)


def evaluate_all(args, failures):
    for spec in specs():
        item = name(spec)
        for mode in ("tf32", "trt_fp32", "trt_fp16"):
            output = OUTPUT_ROOT / "results" / f"{item}_{mode}.json"
            if output.is_file() and not args.force:
                print(f"[eval skip] {item}_{mode}", flush=True)
                continue
            command = [
                sys.executable, str(HERE / "evaluate_backend.py"),
                "--family", spec["family"],
                "--encoder", spec["encoder"],
                "--nominal-resolution", str(spec["resolution"]),
                "--height", str(spec["height"]),
                "--width", str(spec["width"]),
                "--dataset", spec["dataset"],
                "--mode", mode,
                "--checkpoint", str(spec["checkpoint"]),
                "--output", str(output),
            ]
            if spec["family"] == "metric":
                command += ["--metric-domain", spec["domain"]]
            if mode.startswith("trt"):
                engine = OUTPUT_ROOT / "engines" / f"{item}_{mode}.engine"
                if not engine.is_file():
                    failures.append({"stage": "eval", "item": f"{item}_{mode}", "reason": "missing engine"})
                    continue
                command += ["--engine", str(engine), "--plugin", str(PLUGIN)]
            try:
                print(f"[eval] {item}_{mode}", flush=True)
                run(
                    command, ROOT,
                    OUTPUT_ROOT / "logs" / f"eval_{item}_{mode}.log",
                    f"eval {item}_{mode}",
                )
            except Exception as error:
                failures.append({"stage": "eval", "item": f"{item}_{mode}", "reason": str(error)})
                print(f"[eval failed] {item}_{mode}: {error}", flush=True)


def summarize():
    rows = []
    for path in sorted((OUTPUT_ROOT / "results").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append({
            "family": payload["family"],
            "encoder": payload["encoder"],
            "nominal_resolution": payload["nominal_resolution"],
            "input_resolution": f"{payload['input_shape'][0]}x{payload['input_shape'][1]}",
            "dataset": payload["dataset"],
            "checkpoint_domain": payload["checkpoint_domain"],
            "mode": payload["mode"],
            "backend": payload["backend"],
            "delta1": metrics["d1"],
            "abs_rel": metrics.get("abs_rel", ""),
            "rmse": metrics.get("rmse", ""),
            "samples": payload["samples"],
            "evaluation": payload["evaluation"],
            "checkpoint": payload["checkpoint"],
            "engine": payload["engine"] or "",
        })
    expected = {
        (spec["family"], spec["encoder"], spec["resolution"], spec["dataset"], mode)
        for spec in specs()
        for mode in ("tf32", "trt_fp32", "trt_fp16")
    }
    actual = {
        (row["family"], row["encoder"], int(row["nominal_resolution"]), row["dataset"], row["mode"])
        for row in rows
    }
    if actual != expected:
        raise RuntimeError(
            f"incomplete result matrix: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for row in rows:
        expected_samples = 654 if row["dataset"] == "NYUD" else 652
        if int(row["samples"]) != expected_samples:
            raise RuntimeError(f"unexpected sample count: {row}")
    output = OUTPUT_ROOT.parent / "backend_depth_metrics.csv"
    fields = (
        "family", "encoder", "nominal_resolution", "input_resolution", "dataset",
        "checkpoint_domain", "mode", "backend", "delta1", "abs_rel", "rmse",
        "samples", "evaluation", "checkpoint", "engine",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {output} ({len(rows)} rows)", flush=True)
    return len(rows)


def main():
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "export", "build", "eval", "summary"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root.resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    failures = []

    if args.stage in ("all", "export"):
        export_all(args, failures)
    if args.stage in ("all", "build"):
        build_all(args, failures)
    if args.stage in ("all", "eval"):
        evaluate_all(args, failures)
    row_count = summarize() if args.stage in ("all", "summary") else None
    failure_path = OUTPUT_ROOT / "failures.json"
    failure_path.write_text(
        json.dumps({"failures": failures, "failure_count": len(failures)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"failure_count": len(failures), "summary_rows": row_count}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
