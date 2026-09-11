"""Model factory shared by relative-depth inference and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch


MODEL_CHOICES = ("S", "B", "L", "MNV4-S", "MNV4-M", "MNV4-M-SLIM-SPF")


def create_relative_model(encoder: str):
    normalized = encoder.upper()
    if normalized in {"S", "B", "L"}:
        from tinyvim.model.dpt import TinyVimDepth

        return TinyVimDepth(encoder=normalized)

    from mobilenetv4 import MobileNetV4Depth, MobileNetV4MSlimSPFDepth

    if normalized == "MNV4-S":
        return MobileNetV4Depth("S")
    if normalized == "MNV4-M":
        return MobileNetV4Depth("M")
    if normalized == "MNV4-M-SLIM-SPF":
        return MobileNetV4MSlimSPFDepth()
    raise ValueError(f"Unsupported relative-depth encoder: {encoder!r}")


def load_relative_model(
    encoder: str,
    checkpoint: str | Path,
    device: str | torch.device,
):
    checkpoint = Path(checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in state.items()
    }
    model = create_relative_model(encoder)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload
