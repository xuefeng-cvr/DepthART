#!/usr/bin/env python3
"""Export a static-shape DepthART ONNX graph for TensorRT on Orin NX."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


class ExportLayerNorm(nn.Module):
    """LayerNorm decomposition understood by older TensorRT ONNX parsers."""
    def __init__(self, source):
        super().__init__()
        self.normalized_shape = source.normalized_shape
        self.eps = source.eps
        self.weight = source.weight
        self.bias = source.bias

    def forward(self, value):
        axes = tuple(range(value.ndim - len(self.normalized_shape), value.ndim))
        mean = value.mean(dim=axes, keepdim=True)
        centered = value - mean
        variance = centered.square().mean(dim=axes, keepdim=True)
        output = centered * torch.rsqrt(variance + self.eps)
        return output * self.weight + self.bias


def replace_layer_norms(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, ExportLayerNorm(child))
        else:
            replace_layer_norms(child)


def install_exportable_sdpa():
    """Use primitive ONNX ops instead of the fused PyTorch SDPA operator."""
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        factor = scale if scale is not None else query.new_tensor(query.shape[-1] ** -0.5)
        scores = torch.matmul(query, key.transpose(-2, -1)) * factor
        if is_causal:
            causal = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
        attention = torch.softmax(scores, dim=-1)
        if dropout_p:
            attention = F.dropout(attention, dropout_p)
        return torch.matmul(attention, value)
    F.scaled_dot_product_attention = sdpa


class RelativeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)


class MetricWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, K):
        return self.model(image, K).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("relative", "metric"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=("S", "B", "L"), required=True)
    parser.add_argument("--domain", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--simplify", action="store_true", help="Fold static-shape ONNX constants for TensorRT parsers")
    parser.add_argument(
        "--native-layernorm",
        action="store_true",
        help="Preserve ONNX LayerNormalization for TensorRT 10+ FP16 accuracy.",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    image = torch.randn(1, 3, args.height, args.width, device=device)
    if args.family == "relative":
        sys.path.insert(0, str(ROOT / "relative"))
        from tinyvim.model.dpt import TinyVimDepth

        model = TinyVimDepth(encoder=args.encoder)
        payload = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload), strict=True)
        wrapper, inputs, names = RelativeWrapper(model), (image,), ["image"]
    else:
        sys.path.insert(0, str(ROOT / "metric"))
        from model import load_model

        model = load_model(args.checkpoint, args.encoder, args.domain, device)
        K = torch.tensor(
            [[[max(args.width, args.height), 0.0, (args.width - 1) / 2],
              [0.0, max(args.width, args.height), (args.height - 1) / 2],
              [0.0, 0.0, 1.0]]], device=device
        )
        wrapper, inputs, names = MetricWrapper(model), (image, K), ["image", "K"]

    wrapper.to(device).eval()
    if not args.native_layernorm:
        replace_layer_norms(wrapper)
    install_exportable_sdpa()
    for module in wrapper.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.train()  # bypass _native_multi_head_attention during legacy ONNX export
            module.dropout = 0.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sdp_context = (
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        if device == "cuda" else torch.no_grad()
    )
    with torch.inference_mode(), sdp_context:
        torch.onnx.export(
            wrapper,
            inputs,
            output,
            input_names=names,
            output_names=["depth"],
            opset_version=args.opset,
            do_constant_folding=args.family == "relative",
            dynamic_axes=None,
            training=(
                torch.onnx.TrainingMode.EVAL
                if args.family == "relative"
                else torch.onnx.TrainingMode.PRESERVE
            ),
        )
    if args.simplify:
        import onnx
        from onnxsim import simplify
        simplified, check = simplify(onnx.load(str(output)))
        if not check:
            raise RuntimeError("onnxsim numerical/structural check failed")
        onnx.save(simplified, str(output))
    print(f"exported static graph: {output}")
    print(f"inputs: {names}; image shape: [1, 3, {args.height}, {args.width}]")


if __name__ == "__main__":
    main()
