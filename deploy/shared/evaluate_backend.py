#!/usr/bin/env python3
"""Evaluate one DepthART checkpoint/dataset/backend combination."""

import argparse
import ctypes
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = Path(__file__).resolve().parent / "selective_scan"
PLUGIN = SCAN_ROOT / "results" / "plugins" / "libdepthart_selective_scan_trt.so"
DATA_ROOT = Path(os.environ.get("DEPTHART_DATA_ROOT", ROOT / "datasets"))
DATASET_PLACEHOLDER = "/path/to/Zero_shot_Datasets"
sys.path.insert(0, str(SCAN_ROOT))

from depthart_selective_scan import (  # noqa: E402
    disable_unused_auxiliary_features,
    install_depthart,
    parameter_fingerprint,
)


class TensorRTRunner:
    def __init__(self, engine_path, examples, plugin_path):
        import tensorrt as trt

        self.trt = trt
        self.plugin = ctypes.CDLL(str(Path(plugin_path).resolve()), mode=ctypes.RTLD_GLOBAL)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.inputs = {}
        self.outputs = {}

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for name, example in examples.items():
            if name not in names:
                raise RuntimeError(f"engine has no input named {name}: {names}")
            dtype = torch.float16 if self.engine.get_tensor_dtype(name) == trt.float16 else torch.float32
            tensor = torch.empty_like(example, dtype=dtype)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
            self.inputs[name] = tensor

        for name in names:
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = torch.float16 if self.engine.get_tensor_dtype(name) == trt.float16 else torch.float32
            tensor = torch.empty(shape, device="cuda", dtype=dtype)
            self.context.set_tensor_address(name, tensor.data_ptr())
            self.outputs[name] = tensor
        if set(self.outputs) != {"depth"}:
            raise RuntimeError(f"unexpected engine outputs: {sorted(self.outputs)}")

    def __call__(self, **values):
        if set(values) != set(self.inputs):
            raise RuntimeError(f"expected inputs {sorted(self.inputs)}, got {sorted(values)}")
        for name, value in values.items():
            self.inputs[name].copy_(value)
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.outputs["depth"]


def mapped_split(source, old_prefix, new_prefix):
    text = Path(source).read_text(encoding="utf-8").replace(old_prefix, new_prefix)
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


def load_pytorch_model(family, encoder, checkpoint, domain):
    sys.path.insert(0, str(ROOT / family))
    if family == "relative":
        from tinyvim.model import tvimblock
        from tinyvim.model.dpt import TinyVimDepth

        model = TinyVimDepth(encoder=encoder)
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload), strict=True)
        model = model.cuda().eval()
        fingerprint = parameter_fingerprint(model)
        disable_unused_auxiliary_features(model)
    else:
        from model import load_model
        from network import tvimblock

        model = load_model(checkpoint, encoder, domain, "cuda")
        fingerprint = parameter_fingerprint(model)
    install_depthart(tvimblock)
    if parameter_fingerprint(model) != fingerprint:
        raise RuntimeError("backend preparation changed model parameters")
    return model, fingerprint


def evaluate_relative(args, runner):
    sys.path.insert(0, str(ROOT / "relative"))
    from dataset.kitti import get_kitti_loader
    from dataset.nyu2 import get_nyud_loader
    from infer_dataset import (
        DATASET_CONFIGS,
        _build_eval_mask,
        _extract_depth,
        _extract_valid_mask,
    )
    from utils.metric import align_depth_least_square, compute_errors

    dataset_key = "NYU" if args.dataset == "NYUD" else "KITTI"
    config = DATASET_CONFIGS[dataset_key]
    split = ROOT / "relative" / config["splits"][0]
    split = mapped_split(
        split,
        DATASET_PLACEHOLDER,
        str(DATA_ROOT),
    )
    loader_fn = get_nyud_loader if dataset_key == "NYU" else get_kitti_loader
    loader = loader_fn(str(split), size=(args.nominal_resolution, args.nominal_resolution))
    totals = None
    count = 0

    try:
        with torch.inference_mode():
            for sample in loader:
                image = sample["image"].cuda(non_blocking=True).float()
                if tuple(image.shape[-2:]) != (args.height, args.width):
                    raise RuntimeError(f"unexpected input shape {tuple(image.shape[-2:])}")
                depth = _extract_depth(sample)
                valid = _extract_valid_mask(sample, depth)
                pred = runner(image=image) if args.mode.startswith("trt") else runner(image)
                if pred.dim() == 3:
                    pred = pred[:, None]
                pred = F.interpolate(pred.float(), depth.shape[-2:], mode="bilinear", align_corners=True)[0, 0]
                if not torch.isfinite(pred).all():
                    invalid = int((~torch.isfinite(pred)).sum())
                    raise RuntimeError(
                        f"non-finite relative prediction at sample {count}: {invalid} values"
                    )
                pred = pred.cpu().numpy()
                mask = _build_eval_mask(config, depth, valid)
                if not np.any(mask):
                    continue
                pred = align_depth_least_square(pred, depth, mask, config["max_depth"])
                current = compute_errors(depth[mask], pred[mask])
                if totals is None:
                    totals = {key: 0.0 for key in current}
                for key, value in current.items():
                    totals[key] += float(value)
                count += 1
    finally:
        split.unlink(missing_ok=True)
    if not count:
        raise RuntimeError("no valid Relative samples")
    return {key: value / count for key, value in totals.items()}, count


def align_metric_depth(pred, target, mask, min_depth, max_depth):
    """Least-squares scale+shift alignment in metric depth space."""
    x = pred[mask].double()
    y = target[mask].double()
    count = x.numel()
    sum_x = x.sum()
    sum_y = y.sum()
    determinant = count * (x * x).sum() - sum_x * sum_x
    if determinant.abs() <= torch.finfo(torch.float64).eps:
        return pred.clamp(min=min_depth, max=max_depth)
    scale = (count * (x * y).sum() - sum_x * sum_y) / determinant
    shift = (sum_y - scale * sum_x) / count
    return (pred.double() * scale + shift).clamp(
        min=min_depth, max=max_depth
    ).float()


def evaluate_metric(args, runner):
    sys.path.insert(0, str(ROOT / "metric"))
    from common import preprocess
    from infer_dataset import (
        SPECS,
        decode_depth,
        evaluation_mask,
        parse_path_maps,
        read_records,
        sample_K,
    )
    from util.metric import eval_depth

    domain, relative_split, min_depth, max_depth = SPECS[args.dataset]
    split = ROOT / "metric" / relative_split
    records = read_records(
        split, parse_path_maps([f"{DATASET_PLACEHOLDER}={DATA_ROOT}"])
    )
    target_width, target_height = ((640, 480) if domain == "indoor" else (448, 448))
    totals = None
    affine_totals = None
    count = 0

    with torch.inference_mode():
        for parts in records:
            image = cv2.imread(parts[0])
            if image is None:
                raise FileNotFoundError(parts[0])
            depth, valid = decode_depth(args.dataset, parts)
            height, width = image.shape[:2]
            K = sample_K(args.dataset, parts, width, height)
            image_tensor, K_tensor = preprocess(image, K, target_width, target_height)
            image_tensor = image_tensor.cuda()
            K_tensor = K_tensor.cuda()
            if tuple(image_tensor.shape[-2:]) != (args.height, args.width):
                raise RuntimeError(f"unexpected input shape {tuple(image_tensor.shape[-2:])}")
            if args.mode.startswith("trt"):
                pred = runner(image=image_tensor, K=K_tensor)
            else:
                pred = runner(image_tensor, K_tensor)
            if pred.dim() == 3:
                pred = pred[:, None]
            pred = F.interpolate(pred.float(), depth.shape, mode="bilinear", align_corners=True)[0, 0]
            mask = torch.from_numpy(
                evaluation_mask(args.dataset, depth, valid, min_depth, max_depth)
            ).cuda()
            if mask.sum() < 10:
                continue
            target = torch.from_numpy(depth).cuda()
            current = eval_depth(pred[mask], target[mask])
            pred_affine = align_metric_depth(
                pred, target, mask, min_depth, max_depth
            )
            affine_current = eval_depth(pred_affine[mask], target[mask])
            if totals is None:
                totals = {key: 0.0 for key in current}
                affine_totals = {key: 0.0 for key in affine_current}
            for key, value in current.items():
                totals[key] += float(value)
            for key, value in affine_current.items():
                affine_totals[key] += float(value)
            count += 1
    if not count:
        raise RuntimeError("no valid Metric samples")
    return (
        {key: value / count for key, value in totals.items()},
        {key: value / count for key, value in affine_totals.items()},
        count,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("relative", "metric"), required=True)
    parser.add_argument("--encoder", choices=("S", "B", "L"), required=True)
    parser.add_argument("--nominal-resolution", type=int, choices=(224, 448), required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--dataset", choices=("NYUD", "KITTI"), required=True)
    parser.add_argument("--mode", choices=("fp32", "tf32", "trt_fp32", "trt_fp16"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metric-domain", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--engine")
    parser.add_argument("--plugin", default=str(PLUGIN))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode.startswith("trt") and not args.engine:
        parser.error("--engine is required for TensorRT modes")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = args.mode in ("tf32", "trt_fp32")
    torch.backends.cudnn.allow_tf32 = args.mode in ("tf32", "trt_fp32")
    torch.set_float32_matmul_precision("highest" if args.mode == "fp32" else "high")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    if args.mode in ("fp32", "tf32"):
        runner, fingerprint = load_pytorch_model(
            args.family, args.encoder, args.checkpoint, args.metric_domain
        )
        backend = (
            "PyTorch strict FP32 + custom SelectiveScan"
            if args.mode == "fp32"
            else "PyTorch TF32 + custom SelectiveScan"
        )
    else:
        examples = {"image": torch.empty(1, 3, args.height, args.width, device="cuda")}
        if args.family == "metric":
            examples["K"] = torch.empty(1, 3, 3, device="cuda")
        runner = TensorRTRunner(args.engine, examples, args.plugin)
        fingerprint = None
        backend = "TensorRT FP32" if args.mode == "trt_fp32" else "TensorRT FP16"

    if args.family == "relative":
        metrics, samples = evaluate_relative(args, runner)
        affine_metrics = metrics
        evaluation = "affine-invariant disparity scale+shift per image"
    else:
        metrics, affine_metrics, samples = evaluate_metric(args, runner)
        evaluation = "metric depth without alignment; affine depth scale+shift also reported"

    result = {
        "family": args.family,
        "encoder": args.encoder,
        "nominal_resolution": args.nominal_resolution,
        "input_shape": [args.height, args.width],
        "dataset": args.dataset,
        "mode": args.mode,
        "backend": backend,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_domain": args.metric_domain if args.family == "metric" else "relative",
        "engine": str(Path(args.engine).resolve()) if args.engine else None,
        "plugin": str(Path(args.plugin).resolve()) if args.mode.startswith("trt") else None,
        "parameter_sha256": fingerprint,
        "evaluation": evaluation,
        "samples": samples,
        "metrics": metrics,
        "affine_metrics": affine_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
