#!/usr/bin/env python3
"""Evaluate released metric models on indoor and outdoor depth datasets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import make_K, preprocess
from model import load_model
from util.metric import eval_depth


ROOT = Path(__file__).resolve().parent
SPECS = {
    "NYUD": ("indoor", "dataset/splits/nyud/nyud_val.txt", 0.01, 10.0),
    "IBIMS": ("indoor", "dataset/splits/nyud/ibims_val.txt", 0.01, 10.0),
    "SUNRGBD": ("indoor", "dataset/splits/zeroshot/sunrgbd.txt", 0.01, 10.0),
    "DIODE_INDOOR": ("indoor", "dataset/splits/zeroshot/diode_indoor.txt", 0.6, 300.0),
    "KITTI": ("outdoor", "dataset/splits/kitti/val.txt", 0.01, 80.0),
    "DDAD": ("outdoor", "dataset/splits/zeroshot/ddad.txt", 0.01, 80.0),
    "ETH3D_OUTDOOR": ("outdoor", "dataset/splits/zeroshot/eth3d_outdoor.txt", 0.01, 80.0),
    "DIODE_OUTDOOR": ("outdoor", "dataset/splits/zeroshot/diode_outdoor.txt", 0.6, 80.0),
}
METRIC_KEYS = ("d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog")


def parse_path_maps(entries):
    mappings = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"path map must be OLD=NEW, got: {entry}")
        old, new = entry.split("=", 1)
        mappings.append((old.rstrip("/"), new.rstrip("/")))
    return mappings


def remap(path, mappings):
    for old, new in mappings:
        if path == old or path.startswith(old + "/"):
            return new + path[len(old):]
    return path


def read_records(split, mappings):
    records = []
    for line in Path(split).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append([remap(value, mappings) for value in line.split()])
    return records


def read_kitti_K(image_path, width, height):
    image_path = Path(image_path)
    candidates = list(image_path.parents)[:5]
    for parent in candidates:
        calibration = parent / "calib_cam_to_cam.txt"
        if not calibration.is_file():
            continue
        for line in calibration.read_text().splitlines():
            if line.startswith(("P_rect_02:", "P2:")):
                values = [float(x) for x in line.split(":", 1)[1].split()]
                return make_K(values[0], values[5], values[2], values[6])
    return make_K(721.5377, 721.5377, (width - 1) / 2, (height - 1) / 2)


def read_metadata_K(image_path, width, height):
    image_path = Path(image_path)
    candidates = [
        image_path.with_suffix(".json"), image_path.parent / "intrinsics.json",
        image_path.parent / "camera.json", image_path.parent / "metadata.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text())
        for key in ("K", "intrinsics", "camera_matrix"):
            matrix = np.asarray(payload.get(key, []), dtype=np.float32)
            if matrix.shape == (3, 3):
                return matrix
    return make_K(max(width, height), max(width, height), (width - 1) / 2, (height - 1) / 2)


def decode_depth(name, parts):
    depth_path = parts[1]
    if name.startswith("DIODE"):
        depth = np.load(depth_path).astype(np.float32).squeeze()
        mask = np.load(parts[2]).astype(bool).squeeze() if len(parts) > 2 else depth > 0
        return depth, mask
    if name == "SUNRGBD":
        raw = np.asarray(Image.open(depth_path), dtype=np.int16)
        depth = (np.right_shift(raw, 3) | np.left_shift(raw, 13)).astype(np.float32) / 1000.0
        return depth, depth > 0
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(depth_path)
    if name == "NYUD":
        depth = raw.astype(np.float32) / 1000.0
    elif name == "IBIMS":
        depth = raw.astype(np.float32) * 50.0 / 65535.0
    else:
        depth = raw.astype(np.float32) / 256.0
    mask = depth > 0
    if name == "IBIMS":
        invalid_path = depth_path.replace("depth", "mask_invalid")
        invalid = cv2.imread(invalid_path, cv2.IMREAD_UNCHANGED)
        if invalid is not None:
            mask &= invalid > 0
    return depth, mask


def sample_K(name, parts, width, height):
    if name == "NYUD":
        fx = float(parts[2]) if len(parts) > 2 else 525.0
        return make_K(fx, 525.0, 319.5, 239.5)
    if name == "IBIMS":
        return make_K(550.39, 548.55, 319.5, 239.5)
    if name == "KITTI":
        return read_kitti_K(parts[0], width, height)
    if name == "DDAD":
        return make_K(
            2181.53025 * width / 1936.0, 2181.60344 * height / 1216.0,
            928.02188 * width / 1936.0, 615.95678 * height / 1216.0,
        )
    if name.startswith("DIODE"):
        return read_metadata_K(parts[0], width, height)
    if name.startswith("ETH3D") and len(parts) >= 6:
        return make_K(*map(float, parts[2:6]))
    return make_K(548.55, 548.55, (width - 1) / 2, (height - 1) / 2)


def evaluation_mask(name, depth, valid, min_depth, max_depth):
    mask = valid & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    crop = np.ones_like(mask)
    if name == "NYUD":
        crop[:] = False
        crop[45:471, 41:601] = True
    elif name == "KITTI":
        crop[:] = False
        crop[153:371, 44:1197] = True
    return mask & crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=("S", "B", "L"), required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    domain, default_split, min_depth, max_depth = SPECS[args.dataset]
    split = Path(args.split) if args.split else ROOT / default_split
    records = read_records(split, parse_path_maps(args.path_map))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, args.encoder, domain, device)
    target_width, target_height = ((640, 480) if domain == "indoor" else (448, 448))
    totals = {key: 0.0 for key in METRIC_KEYS}
    count = 0

    with torch.inference_mode():
        for parts in records[: args.max_samples or None]:
            image = cv2.imread(parts[0])
            if image is None:
                raise FileNotFoundError(parts[0])
            depth, valid = decode_depth(args.dataset, parts)
            height, width = image.shape[:2]
            K = sample_K(args.dataset, parts, width, height)
            image_tensor, K_tensor = preprocess(image, K, target_width, target_height)
            pred = model(image_tensor.to(device), K_tensor.to(device))[:, None]
            pred = F.interpolate(pred, depth.shape, mode="bilinear", align_corners=True)[0, 0]
            mask = torch.from_numpy(evaluation_mask(args.dataset, depth, valid, min_depth, max_depth)).to(device)
            if mask.sum() < 10:
                continue
            current = eval_depth(pred.float()[mask], torch.from_numpy(depth).to(device)[mask])
            for key in totals:
                totals[key] += current[key]
            count += 1
            if count % 100 == 0:
                print(f"processed {count}/{len(records)}", flush=True)

    if count == 0:
        raise RuntimeError("no valid samples were evaluated")
    metrics = {key: value / count for key, value in totals.items()}
    result = {"dataset": args.dataset, "checkpoint": args.checkpoint, "samples": count, "metrics": metrics}
    print(json.dumps(result, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
