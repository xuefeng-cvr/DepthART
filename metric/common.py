"""Shared image/intrinsics preprocessing for metric inference."""

import cv2
import numpy as np
import torch


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def make_K(fx, fy, cx, cy):
    return np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float32)


def lower_bound_size(width, height, target_width, target_height, multiple=32):
    scale = max(target_width / width, target_height / height)
    def constrain(value, minimum):
        result = int(np.round(value / multiple) * multiple)
        return result if result >= minimum else int(np.ceil(value / multiple) * multiple)
    new_width = constrain(scale * width, target_width)
    new_height = constrain(scale * height, target_height)
    return new_width, new_height


def preprocess(image_bgr, K, target_width, target_height):
    """Keep aspect ratio, lower-bound resize, update K, then ImageNet-normalize."""
    height, width = image_bgr.shape[:2]
    new_width, new_height = lower_bound_size(
        width, height, target_width, target_height
    )
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    K = K.copy()
    K[0, :] *= new_width / width
    K[1, :] *= new_height / height
    image = ((image - MEAN) / STD).transpose(2, 0, 1).copy()
    return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(K).unsqueeze(0)


def colorize(depth):
    valid = np.isfinite(depth) & (depth > 0)
    lo, hi = np.percentile(depth[valid], (2, 98)) if valid.any() else (0.0, 1.0)
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
