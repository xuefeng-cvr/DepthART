#!/usr/bin/env python3
"""Export a static-shape DepthART graph for dataset backend evaluation."""

import argparse
import sys
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = Path(__file__).resolve().parent / "selective_scan"
sys.path.insert(0, str(SCAN_ROOT))

from depthart_selective_scan import (  # noqa: E402
    disable_unused_auxiliary_features,
    install_depthart,
    parameter_fingerprint,
    register_onnx_symbolic,
)


def install_chunked_sdpa(chunk_size):
    """Export exact attention in query chunks to bound TensorRT workspace."""
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        factor = scale if scale is not None else query.new_tensor(query.shape[-1] ** -0.5)
        outputs = []
        query_length = query.shape[-2]
        for start in range(0, query_length, chunk_size):
            end = min(start + chunk_size, query_length)
            query_chunk = query[..., start:end, :]
            scores = torch.matmul(query_chunk, key.transpose(-2, -1)) * factor
            if is_causal:
                query_positions = torch.arange(start, end, device=query.device)[:, None]
                key_positions = torch.arange(key.shape[-2], device=query.device)[None, :]
                scores = scores.masked_fill(key_positions > query_positions, float("-inf"))
            if attn_mask is not None:
                mask = attn_mask[..., start:end, :] if attn_mask.shape[-2] == query_length else attn_mask
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
            attention = torch.softmax(scores, dim=-1)
            if dropout_p:
                attention = F.dropout(attention, dropout_p)
            outputs.append(torch.matmul(attention, value))
        return torch.cat(outputs, dim=-2)

    F.scaled_dot_product_attention = sdpa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("relative", "metric"), required=True)
    parser.add_argument("--encoder", choices=("S", "B", "L"), required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metric-domain", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--attention-chunk-size", type=int, default=0)
    args = parser.parse_args()

    if args.height % 32 or args.width % 32:
        parser.error("height and width must be divisible by 32")

    sys.path.insert(0, str(ROOT / args.family))
    from export_helpers import install_exportable_sdpa

    if args.family == "relative":
        from tinyvim.model import tvimblock
        from tinyvim.model.dpt import TinyVimDepth

        model = TinyVimDepth(encoder=args.encoder)
        payload = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload), strict=True)
        model = model.cuda().eval()
        fingerprint = parameter_fingerprint(model)
        disable_unused_auxiliary_features(model)
        inputs = (torch.randn(1, 3, args.height, args.width, device="cuda"),)
        input_names = ("image",)
    else:
        from model import load_model
        from network import tvimblock

        model = load_model(args.checkpoint, args.encoder, args.metric_domain, "cuda")
        fingerprint = parameter_fingerprint(model)
        image = torch.randn(1, 3, args.height, args.width, device="cuda")
        K = torch.tensor(
            [[[500.0, 0.0, args.width / 2.0],
              [0.0, 500.0, args.height / 2.0],
              [0.0, 0.0, 1.0]]],
            device="cuda",
        )
        inputs = (image, K)
        input_names = ("image", "K")

    install_depthart(tvimblock)
    register_onnx_symbolic(args.opset)
    if args.attention_chunk_size:
        install_chunked_sdpa(args.attention_chunk_size)
    else:
        install_exportable_sdpa()
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module.train()
            module.dropout = 0.0
    if parameter_fingerprint(model) != fingerprint:
        raise RuntimeError("deployment preparation changed model parameters")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            output,
            input_names=input_names,
            output_names=("depth",),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
            training=torch.onnx.TrainingMode.PRESERVE,
        )

    if parameter_fingerprint(model) != fingerprint:
        raise RuntimeError("ONNX export changed model parameters")
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    scans = [
        node for node in graph.graph.node
        if node.domain == "com.depthart" and node.op_type == "SelectiveScan"
    ]
    graph_inputs = {value.name for value in graph.graph.input}
    if not scans:
        raise RuntimeError("export contains no SelectiveScan plugin nodes")
    if not set(input_names).issubset(graph_inputs):
        raise RuntimeError(f"missing graph inputs: {set(input_names) - graph_inputs}")
    print(f"exported={output.resolve()}")
    print(
        f"family={args.family}; encoder={args.encoder}; "
        f"shape={args.height}x{args.width}; inputs={sorted(graph_inputs)}"
    )
    print(f"attention_chunk_size={args.attention_chunk_size}")
    print(f"nodes={len(graph.graph.node)}; scans={len(scans)}")
    print(f"parameter_sha256={fingerprint}")


if __name__ == "__main__":
    main()
