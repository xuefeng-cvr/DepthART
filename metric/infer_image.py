#!/usr/bin/env python3
"""Metric-depth inference for one image."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import colorize, make_K, preprocess
from model import load_model
from network import tvimblock


def enable_optimized_scan():
    extension_root = Path(__file__).resolve().parents[1] / "deploy/shared/selective_scan"
    sys.path.insert(0, str(extension_root))
    try:
        import depthart_selective_scan_cuda  # noqa: F401
        from depthart_selective_scan import install_depthart
    except ImportError:
        return False
    install_depthart(tvimblock)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=("S", "B", "L"), required=True)
    parser.add_argument("--domain", choices=("indoor", "outdoor"), required=True)
    parser.add_argument("--intrinsics", nargs=4, type=float, required=True, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output", default="metric_depth.npy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    raw_height, raw_width = image.shape[:2]
    target_height = args.height or (480 if args.domain == "indoor" else 448)
    target_width = args.width or (640 if args.domain == "indoor" else 448)
    tensor, K = preprocess(image, make_K(*args.intrinsics), target_width, target_height)
    model = load_model(args.checkpoint, args.encoder, args.domain, device)
    optimized_scan = enable_optimized_scan()

    with torch.inference_mode():
        prediction = model(tensor.to(device), K.to(device))[:, None]
        prediction = F.interpolate(
            prediction, (raw_height, raw_width), mode="bilinear", align_corners=True
        )[0, 0]
    depth = prediction.float().cpu().numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, depth)
    cv2.imwrite(str(output.with_suffix(".png")), colorize(depth))
    print(f"saved {output} and {output.with_suffix('.png')} ({depth.shape})")
    print(f"selective_scan={'optimized CUDA extension' if optimized_scan else 'reference fallback'}")


if __name__ == "__main__":
    main()
