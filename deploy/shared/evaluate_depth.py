#!/usr/bin/env python3
"""Run the released Relative and Metric DepthART evaluation matrix."""

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "deploy" / "runs" / "evaluation"
ENCODERS = ("S", "B", "L")
RELATIVE_RESOLUTIONS = (224, 448)
DATA_ROOT = Path(os.environ.get("DEPTHART_DATA_ROOT", ROOT / "datasets"))
DATASET_PLACEHOLDER = "/path/to/Zero_shot_Datasets"


def run_command(command, cwd, log_path):
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        tail = " | ".join(result.stdout.splitlines()[-5:])
        raise RuntimeError(f"exit code {result.returncode}: {tail}")
    return result.stdout


def parse_relative_summary(output, dataset):
    pattern = re.compile(rf"^{re.escape(dataset)}: (\{{.*\}})$", re.MULTILINE)
    match = pattern.search(output)
    if not match:
        raise RuntimeError(f"missing {dataset} summary")
    return ast.literal_eval(match.group(1))


def relative_row(encoder, resolution, dataset, checkpoint, metrics, status, note):
    sample_count = 654 if dataset == "NYU" else 652
    return {
        "family": "relative",
        "encoder": encoder,
        "resolution": f"{resolution}x{resolution}",
        "dataset": "NYUD" if dataset == "NYU" else dataset,
        "checkpoint_domain": "relative",
        "checkpoint": str(checkpoint),
        "evaluation": "affine-invariant disparity scale+shift per image",
        "delta1": metrics.get("d1", ""),
        "abs_rel": metrics.get("abs_rel", ""),
        "rmse": metrics.get("rmse", ""),
        "samples": sample_count if status == "success" else "",
        "status": status,
        "note": note,
    }


def metric_row(encoder, dataset, domain, checkpoint, metrics, samples, status, note):
    return {
        "family": "metric",
        "encoder": encoder,
        "resolution": "640x480 lower-bound" if dataset == "NYUD" else "448x448 lower-bound",
        "dataset": dataset,
        "checkpoint_domain": domain,
        "checkpoint": str(checkpoint),
        "evaluation": "metric depth without alignment",
        "delta1": metrics.get("d1", ""),
        "abs_rel": metrics.get("abs_rel", ""),
        "rmse": metrics.get("rmse", ""),
        "samples": samples if status == "success" else "",
        "status": status,
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / "logs"
    json_dir = output_dir / "json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    relative_map = f"{DATASET_PLACEHOLDER}={DATA_ROOT}"
    for encoder in ENCODERS:
        for resolution in RELATIVE_RESOLUTIONS:
            checkpoint = (
                ROOT / "checkpoints" / "relative"
                / f"depthart_relative_{encoder.lower()}_{resolution}.pth"
            )
            for dataset in ("NYU", "KITTI"):
                name = f"relative_{encoder.lower()}_{resolution}_{dataset.lower()}"
                log_path = logs_dir / f"{name}.log"
                command = [
                    sys.executable,
                    "infer_dataset.py",
                    "--encoder", encoder,
                    "--pretrained_from", str(checkpoint),
                    "--input_height", str(resolution),
                    "--input_width", str(resolution),
                    "--eval_datasets", dataset,
                    "--path-map", relative_map,
                ]
                try:
                    print(f"[run] {name}", flush=True)
                    output = run_command(command, ROOT / "relative", log_path)
                    metrics = parse_relative_summary(output, dataset)
                    rows.append(relative_row(encoder, resolution, dataset, checkpoint, metrics, "success", ""))
                    print(f"[done] {name}: d1={metrics['d1']}, abs_rel={metrics['abs_rel']}", flush=True)
                except Exception as error:
                    rows.append(relative_row(encoder, resolution, dataset, checkpoint, {}, "failed", str(error)))
                    print(f"[failed] {name}: {error}", flush=True)

    metric_map = f"{DATASET_PLACEHOLDER}={DATA_ROOT}"
    kitti_map = metric_map
    for encoder in ENCODERS:
        for dataset, domain, path_map in (
            ("NYUD", "indoor", metric_map),
            ("KITTI", "outdoor", kitti_map),
        ):
            checkpoint = (
                ROOT / "checkpoints" / "metric"
                / f"depthart_metric_{domain}_{encoder.lower()}_448.pth"
            )
            name = f"metric_{encoder.lower()}_{dataset.lower()}"
            log_path = logs_dir / f"{name}.log"
            json_path = json_dir / f"{name}.json"
            command = [
                sys.executable,
                "infer_dataset.py",
                "--dataset", dataset,
                "--encoder", encoder,
                "--checkpoint", str(checkpoint),
                "--path-map", path_map,
                "--output-json", str(json_path),
            ]
            try:
                print(f"[run] {name}", flush=True)
                run_command(command, ROOT / "metric", log_path)
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                metrics = payload["metrics"]
                rows.append(metric_row(
                    encoder, dataset, domain, checkpoint, metrics,
                    int(payload["samples"]), "success", "",
                ))
                print(f"[done] {name}: d1={metrics['d1']:.6f}, rmse={metrics['rmse']:.6f}", flush=True)
            except Exception as error:
                rows.append(metric_row(encoder, dataset, domain, checkpoint, {}, 0, "failed", str(error)))
                print(f"[failed] {name}: {error}", flush=True)

    csv_path = output_dir / "depth_metrics.csv"
    fields = (
        "family", "encoder", "resolution", "dataset", "checkpoint_domain",
        "checkpoint", "evaluation", "delta1", "abs_rel", "rmse", "samples",
        "status", "note",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["status"] != "success"]
    print(f"wrote {csv_path} ({len(rows)} rows, {len(failures)} failed)")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
