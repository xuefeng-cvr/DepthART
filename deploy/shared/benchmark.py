#!/usr/bin/env python3
"""Benchmark the parameter-preserving DepthART deployment path."""

import argparse
import ctypes
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCAN_ROOT = HERE / "selective_scan"
sys.path.insert(0, str(SCAN_ROOT))

from depthart_selective_scan import (  # noqa: E402
    CUDAGraphRunner,
    cache_metric_camera_embeddings,
    cache_metric_daa_attention_kv,
    install_depthart,
    optimize_depthart_inference,
    parameter_fingerprint,
)


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def percentile_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "fps_from_mean": float(1000.0 / values.mean()),
        "raw_ms": values.tolist(),
    }


def gpu_snapshot():
    fields = "name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total,pstate"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        keys = (
            "name", "driver", "temperature_c", "power_w", "sm_clock_mhz", "memory_clock_mhz",
            "memory_used_mib", "memory_total_mib", "pstate",
        )
        return dict(zip(keys, [part.strip() for part in output.split(",")]))
    except Exception as error:
        return {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "nvidia_smi_error": str(error),
        }


def fixed_K(device, resolution):
    return torch.tensor(
        [[
            [500.0 * resolution / 640.0, 0.0, 319.5 * resolution / 640.0],
            [0.0, 500.0 * resolution / 480.0, 239.5 * resolution / 480.0],
            [0.0, 0.0, 1.0],
        ]],
        device=device,
    )


def preprocess(raw_rgb, resolution):
    image = cv2.resize(raw_rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255.0
    image = ((image - MEAN) / STD).transpose(2, 0, 1).copy()
    return torch.from_numpy(image).unsqueeze(0)


def sample_stream(seed, count, height, width):
    """Yield deterministic samples without retaining almost 1 GiB of raw images."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def load_model(family, encoder, checkpoint, domain, device):
    sys.path.insert(0, str(ROOT / family))
    if family == "relative":
        from tinyvim.model import tvimblock
        from tinyvim.model.dpt import TinyVimDepth

        model = TinyVimDepth(encoder=encoder)
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload), strict=True)
    else:
        from model import load_model as load_metric_model
        from network import tvimblock

        model = load_metric_model(checkpoint, encoder, domain, "cpu")
    return model.to(device).eval(), tvimblock


def original_forward(model, family, image, camera):
    with torch.inference_mode():
        return model(image) if family == "relative" else model(image, camera)


def optimize_model(model, tvimblock, family, camera, resolution):
    before = parameter_fingerprint(model)
    if family == "relative":
        optimize_depthart_inference(model, tvimblock)
    else:
        cache_metric_camera_embeddings(model, camera, resolution, resolution)
        cache_metric_daa_attention_kv(model)
        install_depthart(tvimblock)
    if parameter_fingerprint(model) != before:
        raise RuntimeError("optimization changed model parameters")
    return before


class TensorRTRunner:
    def __init__(self, engine_path, plugin_path, example):
        import tensorrt as trt

        self.trt = trt
        self.plugin = ctypes.CDLL(str(Path(plugin_path).resolve()), mode=ctypes.RTLD_GLOBAL)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        self.input_name = next(
            name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_name = next(
            name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        input_dtype = torch.float16 if self.engine.get_tensor_dtype(self.input_name) == trt.float16 else torch.float32
        self.input = torch.empty_like(example, dtype=input_dtype)
        self.context.set_input_shape(self.input_name, tuple(self.input.shape))
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output_dtype = torch.float16 if self.engine.get_tensor_dtype(self.output_name) == trt.float16 else torch.float32
        self.output = torch.empty(output_shape, device=example.device, dtype=output_dtype)
        self.context.set_tensor_address(self.input_name, self.input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())

    def run(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.output


class TRTGraphRunner:
    def __init__(self, runner, example, warmup):
        self.runner = runner
        self.runner.input.copy_(example)
        for _ in range(warmup):
            self.runner.run()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.runner.run()

    def __call__(self, image):
        self.runner.input.copy_(image)
        self.graph.replay()
        return self.output


def compare_outputs(reference, candidate, family):
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    finite = bool(torch.isfinite(candidate).all())
    shape_match = reference.numel() == candidate.numel()
    if not finite or not shape_match:
        return {"finite": finite, "shape_match": shape_match}
    difference = (reference - candidate).abs()
    denominator = reference.abs().mean().clamp_min(1e-8)
    result = {
        "finite": True,
        "shape_match": True,
        "mae": float(difference.mean()),
        "relative_mae": float(difference.mean() / denominator),
        "max_abs": float(difference.max()),
        "cosine": float(torch.nn.functional.cosine_similarity(reference, candidate, dim=0)),
    }
    if family == "relative":
        design = torch.stack((candidate, torch.ones_like(candidate)), dim=1)
        solution = torch.linalg.lstsq(design, reference[:, None]).solution[:, 0]
        aligned = design @ solution
        aligned_difference = (reference - aligned).abs()
        result["affine_aligned_mae"] = float(aligned_difference.mean())
        result["affine_aligned_relative_mae"] = float(aligned_difference.mean() / denominator)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("relative", "metric"), required=True)
    parser.add_argument("--encoder", choices=("S", "B", "L"), required=True)
    parser.add_argument("--resolution", type=int, choices=(224, 448), required=True)
    parser.add_argument("--mode", choices=("fp32", "amp", "trt_fp32", "trt_fp16"), required=True)
    parser.add_argument(
        "--strict-fp32",
        action="store_true",
        help="Disable TF32 for PyTorch FP32 benchmarking.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine")
    parser.add_argument(
        "--plugin",
        default=str(SCAN_ROOT / "results/plugins/libdepthart_selective_scan_trt.so"),
    )
    parser.add_argument("--metric-domain", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--raw-height", type=int, default=480)
    parser.add_argument("--raw-width", type=int, default=640)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode.startswith("trt") and not args.engine:
        parser.error("--engine is required for TensorRT modes")
    if args.strict_fp32 and args.mode != "fp32":
        parser.error("--strict-fp32 is only valid with --mode fp32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tf32 = args.mode in ("fp32", "trt_fp32") and not args.strict_fp32
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    first_raw = next(sample_stream(args.seed, 1, args.raw_height, args.raw_width))
    first_image = preprocess(first_raw, args.resolution).to(device)
    camera = fixed_K(device, args.resolution) if args.family == "metric" else None
    before_gpu = gpu_snapshot()

    model, tvimblock = load_model(args.family, args.encoder, args.checkpoint, args.metric_domain, device)
    fingerprint = parameter_fingerprint(model)
    reference = original_forward(model, args.family, first_image, camera).float().clone()

    if args.mode.startswith("trt"):
        del model
        gc.collect()
        torch.cuda.empty_cache()
        trt_runner = TensorRTRunner(args.engine, args.plugin, first_image)
        runner = TRTGraphRunner(trt_runner, first_image, args.warmup)
        candidate = runner(first_image).float().clone()
        backend = "TensorRT CUDA Graph + SelectiveScan plugin"
    else:
        optimize_model(model, tvimblock, args.family, camera, args.resolution)
        runner = CUDAGraphRunner(model, first_image, amp=args.mode == "amp", warmup=args.warmup)
        candidate = runner(first_image).float().clone()
        backend = "PyTorch CUDA Graph + custom SelectiveScan"
    torch.cuda.synchronize()
    numerical = compare_outputs(reference, candidate, args.family)
    if not numerical.get("finite") or not numerical.get("shape_match"):
        raise RuntimeError(f"numerical validation failed: {numerical}")

    for _ in range(args.warmup):
        runner(first_image)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    model_only = []
    for raw in sample_stream(args.seed, args.samples, args.raw_height, args.raw_width):
        image = preprocess(raw, args.resolution).to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner(image)
        end.record()
        end.synchronize()
        model_only.append(start.elapsed_time(end))

    end_to_end = []
    checksum = 0.0
    for raw in sample_stream(args.seed, args.samples, args.raw_height, args.raw_width):
        started = time.perf_counter_ns()
        image = preprocess(raw, args.resolution).to(device)
        output = runner(image)
        depth = output.float().squeeze().cpu().numpy()
        restored = cv2.resize(depth, (args.raw_width, args.raw_height), interpolation=cv2.INTER_LINEAR)
        checksum += float(restored[0, 0])
        end_to_end.append((time.perf_counter_ns() - started) / 1e6)

    after_gpu = gpu_snapshot()
    try:
        import tensorrt as trt

        trt_version = trt.__version__
    except ImportError:
        trt_version = None
    result = {
        "schema_version": 2,
        "family": args.family,
        "metric_domain": args.metric_domain if args.family == "metric" else None,
        "encoder": args.encoder,
        "resolution": args.resolution,
        "mode": args.mode,
        "backend": backend,
        "samples": args.samples,
        "warmup": args.warmup,
        "seed": args.seed,
        "raw_shape": [args.raw_height, args.raw_width, 3],
        "model_shape": [1, 3, args.resolution, args.resolution],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "parameter_sha256": fingerprint,
        "parameters_unchanged": True,
        "engine": str(Path(args.engine).resolve()) if args.engine else None,
        "plugin": str(Path(args.plugin).resolve()) if args.mode.startswith("trt") else None,
        "model_only_definition": "CUDA events; static graph input D2D copy and inference included; preprocessing/H2D excluded",
        "end_to_end_definition": "wall time: CPU resize/normalize, H2D, graph D2D copy, inference, D2H, resize depth to 640x480",
        "random_generation_included": False,
        "batch_size": 1,
        "tf32": tf32,
        "precision_policy": "strict_fp32" if args.strict_fp32 else args.mode,
        "numerical_vs_original_pytorch_fp32": numerical,
        "numerical_validation_passed": True,
        "model_only": percentile_stats(model_only),
        "end_to_end": percentile_stats(end_to_end),
        "postprocess_checksum": checksum,
        "peak_torch_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "system": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "tensorrt": trt_version,
            "gpu_before": before_gpu,
            "gpu_after": after_gpu,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "family", "encoder", "resolution", "mode", "samples", "warmup", "model_only",
        "end_to_end", "peak_torch_allocated_mib", "numerical_vs_original_pytorch_fp32",
    )}, indent=2))


if __name__ == "__main__":
    main()
