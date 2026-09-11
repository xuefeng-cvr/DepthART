import argparse
import atexit
import json
import os
import tempfile
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.ddad import get_ddad_loader
from dataset.diode import get_diode_loader
from dataset.eth3d import get_eth3d_loader
from dataset.kitti import get_kitti_loader
from dataset.nyu2 import get_nyud_loader
from models import MODEL_CHOICES, load_relative_model
from utils.tiled_kitti import predict_tiled_affine
from utils.metric import RunningAverageDict, align_depth_least_square, compute_errors


RELATIVE_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIGS = {
    "NYU": {
        "splits": ["dataset/splits/val_a800/nyu_val.txt"],
        "loader": get_nyud_loader,
        "max_depth": 10.0,
        "min_depth": 0.01,
        "crop": (slice(45, 471), slice(41, 601)),
    },
    "KITTI": {
        "splits": ["dataset/splits/val_a800/kitti_val.txt"],
        "loader": get_kitti_loader,
        "max_depth": 80.0,
        "min_depth": 0.01,
        "crop": (slice(153, 371), slice(44, 1197)),
    },
    "ETH3D": {
        "splits": ["dataset/splits/val_a800/eth3d_val.txt"],
        "loader": get_eth3d_loader,
        "max_depth": 80.0,
        "min_depth": 0.01,
        "crop": None,
    },
    "DIODE_INDOOR": {
        "splits": ["dataset/splits/val_a800/diode_indoor_val.txt"],
        "loader": get_diode_loader,
        "max_depth": 80.0,
        "min_depth": 0.6,
        "crop": None,
    },
    "DIODE_OUTDOOR": {
        "splits": ["dataset/splits/val_a800/diode_outdoor_val.txt"],
        "loader": get_diode_loader,
        "max_depth": 80.0,
        "min_depth": 0.6,
        "crop": None,
    },
    "DIODE_FULL": {
        "splits": [
            "dataset/splits/val_a800/diode_indoor_val.txt",
            "dataset/splits/val_a800/diode_outdoor_val.txt",
        ],
        "loader": get_diode_loader,
        "max_depth": 80.0,
        "min_depth": 0.6,
        "crop": None,
    },
    "DDAD": {
        "splits": ["dataset/splits/val_a800/ddad_val.txt"],
        "loader": get_ddad_loader,
        "max_depth": 80.0,
        "min_depth": 0.01,
        "crop": None,
    },
}

# Keep the legacy CLI name, but make it match the documented DIODE-Full metric.
DATASET_ALIASES = {"DIODE": "DIODE_FULL"}


def _resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(RELATIVE_ROOT, path_value)


_TEMP_SPLITS = []


def _mapped_split(path, mappings):
    if not mappings:
        return path
    lines = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            for old, new in mappings:
                line = line.replace(old.rstrip("/"), new.rstrip("/"))
            lines.append(line)
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    handle.writelines(lines)
    handle.close()
    _TEMP_SPLITS.append(handle.name)
    return handle.name


@atexit.register
def _remove_temp_splits():
    for path in _TEMP_SPLITS:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_depth(sample):
    return _to_numpy(sample["depth"]).squeeze()


def _extract_valid_mask(sample, depth):
    if "valid_mask" in sample:
        valid_mask = _to_numpy(sample["valid_mask"]).squeeze()
    elif "eval_mask" in sample:
        valid_mask = _to_numpy(sample["eval_mask"]).squeeze()
    else:
        valid_mask = depth > 0
    return valid_mask.astype(bool)


def _build_eval_mask(config, depth, valid_mask):
    eval_mask = valid_mask.copy()
    eval_mask &= depth > config["min_depth"]
    eval_mask &= depth < config["max_depth"]

    if config["crop"] is not None:
        crop_mask = np.zeros_like(depth, dtype=bool)
        crop_mask[config["crop"][0], config["crop"][1]] = True
        eval_mask &= crop_mask

    return eval_mask


def _predict(model, image, tile_width=0, tile_stride=128):
    if not tile_width or image.shape[-1] <= tile_width:
        return model(image)
    prediction = predict_tiled_affine(
        model, image, window_width=tile_width, stride=tile_stride
    )
    return torch.from_numpy(prediction)[None, None].to(image.device)


def evaluate_dataset(
    model,
    device,
    dataset_name,
    split_paths,
    size,
    max_samples=0,
    kitti_tiled=False,
    tile_stride=64,
):
    config = DATASET_CONFIGS[dataset_name]
    metrics = RunningAverageDict()

    with torch.no_grad():
        for split_path in split_paths:
            dataloader = config["loader"](data_dir_root=split_path, size=size)
            description = f"Eval {dataset_name} ({os.path.basename(split_path)})"
            for sample_index, sample in enumerate(tqdm(dataloader, desc=description, leave=False)):
                if max_samples and sample_index >= max_samples:
                    break
                image = sample["image"].to(device).float()
                depth = _extract_depth(sample)
                valid_mask = _extract_valid_mask(sample, depth)

                tile_width = size[0] if dataset_name == "KITTI" and kitti_tiled else 0
                pred = _predict(model, image, tile_width, tile_stride)
                if pred.dim() == 3:
                    pred = pred[:, None]
                pred = F.interpolate(
                    pred, depth.shape[-2:], mode="bilinear", align_corners=True
                )[0, 0]
                pred = pred.detach().cpu().numpy()

                eval_mask = _build_eval_mask(config, depth, valid_mask)
                if not np.any(eval_mask):
                    continue
                pred = align_depth_least_square(
                    pred, depth, eval_mask, config["max_depth"]
                )
                metrics.update(compute_errors(depth[eval_mask], pred[eval_mask]))

    return OrderedDict(
        (key, round(value, 3)) for key, value in metrics.get_value().items()
    )


def _default_split(dataset_name, index=0):
    return DATASET_CONFIGS[dataset_name]["splits"][index]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate released DepthART relative-depth checkpoints."
    )
    parser.add_argument("--encoder", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--pretrained_from", required=True)
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["NYU", "KITTI"],
        choices=list(DATASET_CONFIGS) + list(DATASET_ALIASES),
    )
    parser.add_argument("--input_height", type=int, default=448)
    parser.add_argument("--input_width", type=int, default=448)
    parser.add_argument("--nyu_eval_fileslist", default=_default_split("NYU"))
    parser.add_argument("--kitti_eval_fileslist", default=_default_split("KITTI"))
    parser.add_argument("--eth3d_eval_fileslist", default=_default_split("ETH3D"))
    parser.add_argument(
        "--diode_indoor_eval_fileslist", default=_default_split("DIODE_INDOOR")
    )
    parser.add_argument(
        "--diode_outdoor_eval_fileslist", default=_default_split("DIODE_OUTDOOR")
    )
    parser.add_argument(
        "--diode_eval_fileslist",
        default=None,
        help="Legacy override: evaluate this one split when requesting DIODE.",
    )
    parser.add_argument("--ddad_eval_fileslist", default=_default_split("DDAD"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--kitti-tiled",
        action="store_true",
        help=(
            "Use affine-consistent square windows for KITTI: jointly align tiles "
            "from overlaps, then blend with a Hann window."
        ),
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=None,
        help="Horizontal KITTI stride (default: 128 at 448, 64 at 224).",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--path-map", action="append", default=[], metavar="OLD=NEW",
        help="Remap absolute prefixes stored in legacy split files; repeat as needed.",
    )
    args = parser.parse_args()
    tile_stride = (
        args.tile_stride
        if args.tile_stride is not None
        else max(1, args.input_width * 2 // 7)
    )
    if args.kitti_tiled and not 0 < tile_stride < args.input_width:
        parser.error("--tile-stride must be between 1 and input_width - 1")

    mappings = []
    for entry in args.path_map:
        if "=" not in entry:
            parser.error(f"--path-map must be OLD=NEW, got {entry}")
        mappings.append(tuple(entry.split("=", 1)))

    split_paths = {
        "NYU": [args.nyu_eval_fileslist],
        "KITTI": [args.kitti_eval_fileslist],
        "ETH3D": [args.eth3d_eval_fileslist],
        "DIODE_INDOOR": [args.diode_indoor_eval_fileslist],
        "DIODE_OUTDOOR": [args.diode_outdoor_eval_fileslist],
        "DIODE_FULL": [
            args.diode_indoor_eval_fileslist,
            args.diode_outdoor_eval_fileslist,
        ],
        "DDAD": [args.ddad_eval_fileslist],
    }
    split_paths = {
        name: [_mapped_split(_resolve_path(path), mappings) for path in paths]
        for name, paths in split_paths.items()
    }
    checkpoint_path = _resolve_path(args.pretrained_from)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, checkpoint = load_relative_model(args.encoder, checkpoint_path, device)
    print("Checkpoint loaded with strict=True")
    print(f"Loading model {checkpoint_path}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("epoch =", checkpoint["epoch"])

    size = (args.input_width, args.input_height)
    summary = OrderedDict()
    for requested_name in args.eval_datasets:
        dataset_name = DATASET_ALIASES.get(requested_name, requested_name)
        requested_splits = split_paths[dataset_name]
        if requested_name == "DIODE" and args.diode_eval_fileslist:
            requested_splits = [_resolve_path(args.diode_eval_fileslist)]
        summary[requested_name] = evaluate_dataset(
            model=model,
            device=device,
            dataset_name=dataset_name,
            split_paths=requested_splits,
            size=size,
            max_samples=args.max_samples,
            kitti_tiled=args.kitti_tiled,
            tile_stride=tile_stride,
        )

    print("\n=== Relative Depth Evaluation Summary ===")
    for dataset_name, metrics in summary.items():
        print(f"{dataset_name}: {dict(metrics)}")
    if args.output_json:
        output = {
            "encoder": args.encoder,
            "checkpoint": checkpoint_path,
            "input_resolution": [args.input_height, args.input_width],
            "affine_alignment": "per-image least-squares scale and shift",
            "kitti_tiled": args.kitti_tiled,
            "tile_stride": tile_stride if args.kitti_tiled else None,
            "tile_alignment": (
                "joint overlap affine alignment with Hann blending"
                if args.kitti_tiled
                else None
            ),
            "max_samples": args.max_samples,
            "metrics": summary,
        }
        output_path = _resolve_path(args.output_json)
        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
            handle.write("\n")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
