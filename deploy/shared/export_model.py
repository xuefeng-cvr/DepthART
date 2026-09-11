#!/usr/bin/env python3
"""Export static DepthART ONNX graphs."""

import argparse
import sys
from pathlib import Path

import onnx
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCAN_ROOT = HERE / "selective_scan"
sys.path.insert(0, str(SCAN_ROOT))

from depthart_selective_scan import (  # noqa: E402
    cache_metric_camera_embeddings,
    cache_metric_daa_attention_kv,
    disable_unused_auxiliary_features,
    install_depthart,
    parameter_fingerprint,
    register_onnx_symbolic,
)


def fixed_K(device, resolution):
    scale_x = resolution / 640.0
    scale_y = resolution / 480.0
    return torch.tensor(
        [[
            [500.0 * scale_x, 0.0, 319.5 * scale_x],
            [0.0, 500.0 * scale_y, 239.5 * scale_y],
            [0.0, 0.0, 1.0],
        ]],
        device=device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("relative", "metric"), required=True)
    parser.add_argument(
        "--encoder",
        choices=("S", "B", "L", "MNV4-S", "MNV4-M", "MNV4-M-SLIM-SPF"),
        required=True,
    )
    parser.add_argument("--resolution", type=int, choices=(224, 448), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metric-domain", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--cache-daa-kv", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.family == "relative" and args.cache_daa_kv:
        parser.error("--cache-daa-kv applies only to metric models")
    if args.family == "metric" and args.encoder.startswith("MNV4"):
        parser.error("MobileNetV4 encoders are available only for relative depth")

    sys.path.insert(0, str(ROOT / args.family))
    from export_helpers import install_exportable_sdpa

    uses_selective_scan = not args.encoder.startswith("MNV4")
    tvimblock = None
    if args.family == "relative":
        from models import load_relative_model

        model, _ = load_relative_model(args.encoder, args.checkpoint, args.device)
        fingerprint = parameter_fingerprint(model)
        if uses_selective_scan:
            from tinyvim.model import tvimblock

            disable_unused_auxiliary_features(model)
    else:
        from model import load_model
        from network import tvimblock

        model = load_model(args.checkpoint, args.encoder, args.metric_domain, args.device)
        fingerprint = parameter_fingerprint(model)
        camera = fixed_K(args.device, args.resolution)
        cache_metric_camera_embeddings(model, camera, args.resolution, args.resolution)
        if args.cache_daa_kv:
            cache_metric_daa_attention_kv(model)

    if uses_selective_scan:
        install_depthart(tvimblock)
        register_onnx_symbolic(args.opset)
    install_exportable_sdpa()
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module.train()
            module.dropout = 0.0
    if parameter_fingerprint(model) != fingerprint:
        raise RuntimeError("deployment preparation changed model parameters")

    image = torch.randn(1, 3, args.resolution, args.resolution, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (image,),
            output,
            input_names=("image",),
            output_names=("depth",),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
            training=(
                torch.onnx.TrainingMode.PRESERVE
                if uses_selective_scan
                else torch.onnx.TrainingMode.EVAL
            ),
        )

    if parameter_fingerprint(model) != fingerprint:
        raise RuntimeError("ONNX export changed model parameters")
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    scans = [
        node for node in graph.graph.node
        if node.domain == "com.depthart" and node.op_type == "SelectiveScan"
    ]
    if uses_selective_scan and not scans:
        raise RuntimeError("export contains no SelectiveScan plugin nodes")
    if not uses_selective_scan and scans:
        raise RuntimeError("MobileNetV4 export unexpectedly contains SelectiveScan nodes")
    print(f"exported={output.resolve()}")
    print(f"family={args.family}; encoder={args.encoder}; resolution={args.resolution}")
    print(
        f"cache_daa_kv={args.cache_daa_kv}; nodes={len(graph.graph.node)}; "
        f"scans={len(scans)}; standard_onnx={not uses_selective_scan}"
    )
    print(f"parameter_sha256={fingerprint}")


if __name__ == "__main__":
    main()
