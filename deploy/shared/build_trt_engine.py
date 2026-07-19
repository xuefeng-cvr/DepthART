#!/usr/bin/env python3
"""Build a full DepthART TensorRT engine with the SelectiveScan plugin."""

import argparse
import ctypes
from pathlib import Path

import tensorrt as trt


def load_plugin(path):
    return ctypes.CDLL(str(Path(path).resolve()), mode=ctypes.RTLD_GLOBAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", choices=("fp32", "tf32", "fp16"), required=True)
    parser.add_argument("--timing-cache", required=True)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--normalization-fp32", action="store_true")
    parser.add_argument(
        "--output-head-fp32",
        action="store_true",
        help="keep the final depth prediction convolutions in FP32 to prevent FP16 overflow",
    )
    args = parser.parse_args()

    plugin = load_plugin(args.plugin)
    plugin.depthart_selective_scan_trt_version.restype = ctypes.c_char_p
    print(f"loaded plugin: {plugin.depthart_selective_scan_trt_version().decode()}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(Path(args.onnx).resolve())):
        errors = "\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors))
        raise RuntimeError(errors)

    constrained_normalizations = 0
    constrained_output_head = 0
    if args.normalization_fp32 or args.output_head_fp32:
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            force_normalization = (
                args.normalization_fp32 and layer.type == trt.LayerType.NORMALIZATION
            )
            force_output_head = args.output_head_fp32 and (
                layer.name.startswith("/depth_head/output_conv1/")
                or layer.name.startswith("/depth_head/output_conv2/")
            )
            if not force_normalization and not force_output_head:
                continue
            layer.precision = trt.float32
            for output_index in range(layer.num_outputs):
                output = layer.get_output(output_index)
                if output.dtype in (trt.float16, trt.float32):
                    layer.set_output_type(output_index, trt.float32)
            constrained_normalizations += int(force_normalization)
            constrained_output_head += int(force_output_head)

    if args.output_head_fp32 and constrained_output_head != 5:
        raise RuntimeError(
            f"expected 5 final depth-head layers, found {constrained_output_head}; "
            "check the exported ONNX layer names"
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    if args.precision != "tf32" and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    if args.normalization_fp32 or args.output_head_fp32:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    cache_path = Path(args.timing_cache)
    cache = config.create_timing_cache(cache_path.read_bytes() if cache_path.is_file() else b"")
    config.set_timing_cache(cache, ignore_mismatch=True)
    print(
        f"precision={args.precision}; normalization_fp32={args.normalization_fp32}; "
        f"output_head_fp32={args.output_head_fp32}; "
        f"constrained_normalizations={constrained_normalizations}; "
        f"constrained_output_head={constrained_output_head}"
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(config.get_timing_cache().serialize())
    print(f"built: {output.resolve()} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
