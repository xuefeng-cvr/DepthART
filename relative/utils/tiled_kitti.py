"""Affine-consistent sliding-window inference for wide KITTI images."""

from __future__ import annotations

import numpy as np
import torch


def sliding_window_starts(image_width, window_width, stride):
    if window_width <= 0 or stride <= 0:
        raise ValueError("window_width and stride must be positive")
    if image_width < window_width:
        raise ValueError(
            "image width {} is smaller than window width {}".format(
                image_width, window_width
            )
        )

    starts = list(range(0, image_width - window_width + 1, stride))
    final_start = image_width - window_width
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def joint_affine_align(
    tiles,
    starts,
    sample_stride=4,
    minimum_overlap=16,
    trim_quantile=0.9,
):
    """Solve one affine transform per tile using pairwise overlaps only."""
    if tiles.ndim != 3:
        raise ValueError("tiles must have shape [N,H,W], got {}".format(tiles.shape))
    if tiles.shape[0] != len(starts):
        raise ValueError("the number of tiles and starts must match")
    if len(starts) == 1:
        return tiles.astype(np.float32, copy=True)

    tile_count = len(starts)
    window_width = tiles.shape[-1]
    anchor = tile_count // 2
    variable_tiles = [index for index in range(tile_count) if index != anchor]
    variable_index = {
        tile_index: 2 * index for index, tile_index in enumerate(variable_tiles)
    }
    rows = []
    targets = []

    for left_index in range(tile_count):
        for right_index in range(left_index + 1, tile_count):
            overlap_left = max(starts[left_index], starts[right_index])
            overlap_right = min(
                starts[left_index] + window_width,
                starts[right_index] + window_width,
            )
            if overlap_right - overlap_left < minimum_overlap:
                continue

            left_values = tiles[
                left_index,
                ::sample_stride,
                overlap_left
                - starts[left_index] : overlap_right
                - starts[left_index] : sample_stride,
            ].reshape(-1).astype(np.float64)
            right_values = tiles[
                right_index,
                ::sample_stride,
                overlap_left
                - starts[right_index] : overlap_right
                - starts[right_index] : sample_stride,
            ].reshape(-1).astype(np.float64)
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            left_values = left_values[finite]
            right_values = right_values[finite]
            if left_values.size < 32:
                continue

            matrix = np.zeros(
                (left_values.size, 2 * (tile_count - 1)), dtype=np.float64
            )
            constant = np.zeros(left_values.size, dtype=np.float64)
            if left_index == anchor:
                constant += left_values
            else:
                column = variable_index[left_index]
                matrix[:, column] = left_values
                matrix[:, column + 1] = 1.0
            if right_index == anchor:
                constant -= right_values
            else:
                column = variable_index[right_index]
                matrix[:, column] -= right_values
                matrix[:, column + 1] -= 1.0
            rows.append(matrix)
            targets.append(-constant)

    if not rows:
        return tiles.astype(np.float32, copy=True)

    matrix = np.concatenate(rows)
    target = np.concatenate(targets)
    coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]

    residual = np.abs(matrix @ coefficients - target)
    keep = residual <= np.quantile(residual, trim_quantile)
    if keep.sum() >= matrix.shape[1] * 4:
        coefficients = np.linalg.lstsq(matrix[keep], target[keep], rcond=None)[0]

    aligned_tiles = []
    for tile_index, tile in enumerate(tiles):
        if tile_index == anchor:
            scale, shift = 1.0, 0.0
        else:
            column = variable_index[tile_index]
            scale, shift = coefficients[column], coefficients[column + 1]
        aligned_tiles.append(scale * tile + shift)
    return np.stack(aligned_tiles).astype(np.float32)


def hann_blend(tiles, starts, image_width):
    """Blend aligned tiles using a horizontal Hann window."""
    window_width = tiles.shape[-1]
    weight = np.hanning(window_width + 2)[1:-1].astype(np.float32)
    output = np.zeros((tiles.shape[1], image_width), dtype=np.float32)
    denominator = np.zeros((1, image_width), dtype=np.float32)
    for tile, start in zip(tiles, starts):
        output[:, start : start + window_width] += tile * weight[None]
        denominator[:, start : start + window_width] += weight[None]
    return output / np.maximum(denominator, 1e-7)


def predict_tiled_affine(model, image, window_width=448, stride=128):
    """Predict a batch-size-one image without using GT during stitching."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape [1,C,H,W], got {}".format(image.shape))
    image_width = int(image.shape[-1])
    starts = sliding_window_starts(image_width, window_width, stride)
    tile_batch = torch.cat(
        [image[:, :, :, start : start + window_width] for start in starts], dim=0
    )
    prediction = model(tile_batch).float()
    if prediction.ndim == 4:
        prediction = prediction[:, 0]
    elif prediction.ndim != 3:
        raise ValueError(
            "model prediction must have 3 or 4 dimensions, got {}".format(
                prediction.shape
            )
        )
    tiles = prediction.detach().cpu().numpy()
    return hann_blend(joint_affine_align(tiles, starts), starts, image_width)
