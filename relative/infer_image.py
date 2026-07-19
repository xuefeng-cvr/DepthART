#!/usr/bin/env python3
"""Affine-invariant relative-depth inference for one image."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tinyvim.model import tvimblock
from tinyvim.model.dpt import TinyVimDepth


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def enable_optimized_scan(model):
    extension_root = Path(__file__).resolve().parents[1] / "deploy/shared/selective_scan"
    sys.path.insert(0, str(extension_root))
    try:
        import depthart_selective_scan_cuda  # noqa: F401
        from depthart_selective_scan import disable_unused_auxiliary_features, install_depthart
    except ImportError:
        return False
    disable_unused_auxiliary_features(model)
    install_depthart(tvimblock)
    return True


def colorize(depth):
    lo, hi = np.percentile(depth[np.isfinite(depth)], (2, 98))
    value = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=("S", "B", "L"), required=True)
    parser.add_argument("--resolution", choices=(224, 448), type=int, required=True)
    parser.add_argument("--output", default="relative_depth.npy")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    raw_height, raw_width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (args.resolution, args.resolution), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy()).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyVimDepth(encoder=args.encoder)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.to(device).eval()
    optimized_scan = enable_optimized_scan(model)
    with torch.inference_mode():
        prediction = model(tensor.to(device))
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
