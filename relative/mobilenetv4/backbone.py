"""Self-contained MobileNetV4-Conv-S/M feature encoders.

The training environment intentionally keeps the old ``timm==0.6.12`` needed
by the original TinyViM code.  MobileNetV4 was added to timm much later, so
this module implements only the two convolutional variants used by DepthART.
Its module names and tensor shapes match timm's ImageNet checkpoints.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class ConvBnAct(nn.Module):
    """Conv-BN-ReLU with names compatible with timm's ConvBnAct."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act1(self.bn1(self.conv(inputs)))


class _NamedConvBnAct(nn.Module):
    """Conv-BN-(ReLU) used inside UIB with timm-compatible ``bn`` names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        groups: int = 1,
        activate: bool,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(inputs)))


class EdgeResidual(nn.Module):
    """Fused inverted bottleneck used at the start of Conv-M."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
    ) -> None:
        super().__init__()
        expanded_channels = _make_divisible(in_channels * expand_ratio)
        self.conv_exp = nn.Conv2d(
            in_channels,
            expanded_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_pwl = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.has_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.act1(self.bn1(self.conv_exp(inputs)))
        outputs = self.bn2(self.conv_pwl(outputs))
        return outputs + inputs if self.has_residual else outputs


class UniversalInvertedResidual(nn.Module):
    """Universal inverted bottleneck (UIB) from MobileNetV4."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        start_dw_kernel_size: int,
        middle_dw_kernel_size: int,
        stride: int,
        expand_ratio: float,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"UIB stride must be 1 or 2, got {stride}")
        if stride == 2 and not (start_dw_kernel_size or middle_dw_kernel_size):
            raise ValueError("A downsampling UIB requires a depthwise convolution")

        stride_in_middle = middle_dw_kernel_size > 0
        if start_dw_kernel_size:
            self.dw_start = _NamedConvBnAct(
                in_channels,
                in_channels,
                start_dw_kernel_size,
                stride=1 if stride_in_middle else stride,
                groups=in_channels,
                activate=False,
            )
        else:
            self.dw_start = nn.Identity()

        expanded_channels = (
            _make_divisible(in_channels * expand_ratio)
            if hidden_channels is None
            else int(hidden_channels)
        )
        if expanded_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {expanded_channels}")
        self.pw_exp = _NamedConvBnAct(
            in_channels,
            expanded_channels,
            1,
            activate=True,
        )
        if middle_dw_kernel_size:
            self.dw_mid = _NamedConvBnAct(
                expanded_channels,
                expanded_channels,
                middle_dw_kernel_size,
                stride=stride,
                groups=expanded_channels,
                activate=True,
            )
        else:
            self.dw_mid = nn.Identity()
        self.pw_proj = _NamedConvBnAct(
            expanded_channels,
            out_channels,
            1,
            activate=False,
        )
        self.has_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dw_start(inputs)
        outputs = self.pw_exp(outputs)
        outputs = self.dw_mid(outputs)
        outputs = self.pw_proj(outputs)
        return outputs + inputs if self.has_residual else outputs


_SMALL_STAGE_SPECS = (
    (("conv", 3, 2, 32), ("conv", 1, 1, 32)),
    (("conv", 3, 2, 96), ("conv", 1, 1, 64)),
    (
        ("uib", 5, 5, 2, 96, 3.0),
        ("uib", 0, 3, 1, 96, 2.0),
        ("uib", 0, 3, 1, 96, 2.0),
        ("uib", 0, 3, 1, 96, 2.0),
        ("uib", 0, 3, 1, 96, 2.0),
        ("uib", 3, 0, 1, 96, 4.0),
    ),
    (
        ("uib", 3, 3, 2, 128, 6.0),
        ("uib", 5, 5, 1, 128, 4.0),
        ("uib", 0, 5, 1, 128, 4.0),
        ("uib", 0, 5, 1, 128, 3.0),
        ("uib", 0, 3, 1, 128, 4.0),
        ("uib", 0, 3, 1, 128, 4.0),
    ),
    (("conv", 1, 1, 960),),
)


_MEDIUM_STAGE_SPECS = (
    (("edge", 3, 2, 48, 4.0),),
    (
        ("uib", 3, 5, 2, 80, 4.0),
        ("uib", 3, 3, 1, 80, 2.0),
    ),
    (
        ("uib", 3, 5, 2, 160, 6.0),
        ("uib", 3, 3, 1, 160, 4.0),
        ("uib", 3, 3, 1, 160, 4.0),
        ("uib", 3, 5, 1, 160, 4.0),
        ("uib", 3, 3, 1, 160, 4.0),
        ("uib", 3, 0, 1, 160, 4.0),
        ("uib", 0, 0, 1, 160, 2.0),
        ("uib", 3, 0, 1, 160, 4.0),
    ),
    (
        ("uib", 5, 5, 2, 256, 6.0),
        ("uib", 5, 5, 1, 256, 4.0),
        ("uib", 3, 5, 1, 256, 4.0),
        ("uib", 3, 5, 1, 256, 4.0),
        ("uib", 0, 0, 1, 256, 4.0),
        ("uib", 3, 0, 1, 256, 4.0),
        ("uib", 3, 5, 1, 256, 2.0),
        ("uib", 5, 5, 1, 256, 4.0),
        ("uib", 0, 0, 1, 256, 4.0),
        ("uib", 0, 0, 1, 256, 4.0),
        ("uib", 5, 0, 1, 256, 2.0),
    ),
    (("conv", 1, 1, 960),),
)


# MobileNetV4-M-slim preserves every spatial block and every public stage
# width from Conv-M.  Only the private UIB expansion channels in the 1/16 and
# 1/32 stages are reduced.  The final 256->960 classification pre-head is not
# part of the SPF pyramid and is intentionally omitted by the slim feature
# extractor below.
_MEDIUM_SLIM_STAGE_SPECS = (
    _MEDIUM_STAGE_SPECS[0],
    _MEDIUM_STAGE_SPECS[1],
    (
        ("uib", 3, 5, 2, 160, 6.0, 320),
        ("uib", 3, 3, 1, 160, 4.0, 352),
        ("uib", 3, 3, 1, 160, 4.0, 352),
        ("uib", 3, 5, 1, 160, 4.0, 352),
        ("uib", 3, 3, 1, 160, 4.0, 352),
        ("uib", 3, 0, 1, 160, 4.0, 352),
        ("uib", 0, 0, 1, 160, 2.0, 320),
        ("uib", 3, 0, 1, 160, 4.0, 352),
    ),
    (
        ("uib", 5, 5, 2, 256, 6.0, 640),
        ("uib", 5, 5, 1, 256, 4.0, 928),
        ("uib", 3, 5, 1, 256, 4.0, 928),
        ("uib", 3, 5, 1, 256, 4.0, 928),
        ("uib", 0, 0, 1, 256, 4.0, 928),
        ("uib", 3, 0, 1, 256, 4.0, 928),
        ("uib", 3, 5, 1, 256, 2.0, 512),
        ("uib", 5, 5, 1, 256, 4.0, 928),
        ("uib", 0, 0, 1, 256, 4.0, 928),
        ("uib", 0, 0, 1, 256, 4.0, 928),
        ("uib", 5, 0, 1, 256, 2.0, 512),
    ),
)


_VARIANT_ALIASES = {
    "s": "small",
    "small": "small",
    "conv-s": "small",
    "conv_small": "small",
    "mobilenetv4-conv-s": "small",
    "mobilenetv4_conv_small": "small",
    "m": "medium",
    "medium": "medium",
    "conv-m": "medium",
    "conv_medium": "medium",
    "mobilenetv4-conv-m": "medium",
    "mobilenetv4_conv_medium": "medium",
}


class MobileNetV4ConvFeatures(nn.Module):
    """MobileNetV4 pyramid returning stride 4, 8, 16 and 32 features."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        normalized = _VARIANT_ALIASES.get(str(variant).strip().lower())
        if normalized is None:
            choices = ", ".join(sorted(_VARIANT_ALIASES))
            raise ValueError(f"Unsupported MobileNetV4 variant {variant!r}; choose one of: {choices}")
        self.variant = normalized
        self.out_channels = [32, 64, 96, 960] if normalized == "small" else [48, 80, 160, 960]

        self.conv_stem = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)

        specs = _SMALL_STAGE_SPECS if normalized == "small" else _MEDIUM_STAGE_SPECS
        stages = []
        in_channels = 32
        for stage_specs in specs:
            blocks = []
            for spec in stage_specs:
                kind, *values = spec
                if kind == "conv":
                    kernel_size, stride, out_channels = values
                    block = ConvBnAct(in_channels, out_channels, kernel_size, stride)
                elif kind == "edge":
                    kernel_size, stride, out_channels, expand_ratio = values
                    block = EdgeResidual(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        expand_ratio,
                    )
                elif kind == "uib":
                    start_kernel, middle_kernel, stride, out_channels, expand_ratio, *hidden = values
                    if len(hidden) > 1:
                        raise ValueError(f"Invalid UIB stage specification: {spec}")
                    block = UniversalInvertedResidual(
                        in_channels,
                        out_channels,
                        start_kernel,
                        middle_kernel,
                        stride,
                        expand_ratio,
                        hidden_channels=hidden[0] if hidden else None,
                    )
                else:  # pragma: no cover - guarded by the constant architecture specs
                    raise AssertionError(f"Unknown block type: {kind}")
                blocks.append(block)
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.blocks = nn.ModuleList(stages)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.act1(self.bn1(self.conv_stem(inputs)))
        pyramid = []
        for stage_index, stage in enumerate(self.blocks):
            outputs = stage(outputs)
            if stage_index in (0, 1, 2, 4):
                pyramid.append(outputs)
        return pyramid

    def load_timm_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        """Load a timm classification checkpoint while ignoring its classifier."""
        prefixes: Iterable[str] = ("module.pretrained.", "pretrained.", "module.")
        normalized = {}
        for name, tensor in state_dict.items():
            clean_name = name
            for prefix in prefixes:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
                    break
            normalized[clean_name] = tensor

        own_keys = set(self.state_dict())
        backbone_state = {name: tensor for name, tensor in normalized.items() if name in own_keys}
        result = self.load_state_dict(backbone_state, strict=False)
        missing = list(result.missing_keys)
        ignored = [name for name in normalized if name not in own_keys]
        if missing:
            raise RuntimeError(
                "MobileNetV4 checkpoint does not cover the complete encoder; "
                f"missing {len(missing)} tensors (first: {missing[0]})"
            )
        return missing, ignored


class MobileNetV4MSlimFeatures(MobileNetV4ConvFeatures):
    """Conv-M hierarchy with structured UIB expansion slimming for SPF."""

    def __init__(self) -> None:
        # Construct the official M stem and early stages first, then replace
        # the deep stages with their width-slimmed, topology-identical forms.
        super().__init__("M")
        stages = [self.blocks[0], self.blocks[1]]
        in_channels = 80
        for stage_specs in _MEDIUM_SLIM_STAGE_SPECS[2:]:
            blocks = []
            for spec in stage_specs:
                kind, *values = spec
                if kind != "uib":
                    raise AssertionError(f"Unexpected M-slim block type: {kind}")
                start_kernel, middle_kernel, stride, out_channels, expand_ratio, hidden_channels = values
                blocks.append(
                    UniversalInvertedResidual(
                        in_channels,
                        out_channels,
                        start_kernel,
                        middle_kernel,
                        stride,
                        expand_ratio,
                        hidden_channels=hidden_channels,
                    )
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.blocks = nn.ModuleList(stages)
        self.variant = "medium_slim"
        self.out_channels = [48, 80, 160, 256]
        self._initialize_weights()

    def forward_with_stem(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        stem = self.act1(self.bn1(self.conv_stem(inputs)))
        outputs = stem
        pyramid = []
        for stage in self.blocks:
            outputs = stage(outputs)
            pyramid.append(outputs)
        return stem, pyramid

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_with_stem(inputs)[1]

    @staticmethod
    def _normalized_source_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized = {}
        for name, tensor in state_dict.items():
            clean_name = name
            for prefix in ("module.pretrained.", "pretrained.", "module."):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
                    break
            if clean_name.startswith(("conv_stem.", "bn1.", "blocks.")):
                normalized[clean_name] = tensor
        return normalized

    @staticmethod
    def _channel_indices(
        source: dict[str, torch.Tensor],
        block_prefix: str,
        target_channels: int,
    ) -> torch.Tensor:
        projection = source[f"{block_prefix}.pw_proj.conv.weight"].float()
        middle_bn = f"{block_prefix}.dw_mid.bn.weight"
        expansion_bn = f"{block_prefix}.pw_exp.bn.weight"
        gamma = source[middle_bn if middle_bn in source else expansion_bn].float().abs()
        importance = projection.abs().mean(dim=(0, 2, 3)) * gamma
        selected = torch.topk(importance, k=target_channels, largest=True, sorted=False).indices
        return selected.sort().values

    def load_structured_m_checkpoint(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        """Prune a trained Conv-M checkpoint into the slim expansion tensors.

        A single importance ranking is used consistently for pw-expansion,
        its BN, the optional depthwise mixer and pw-projection input channels.
        Public stage tensors are copied exactly.
        """
        source = self._normalized_source_state(state_dict)
        target = self.state_dict()
        selected_by_block: dict[str, torch.Tensor] = {}
        for stage_index in (2, 3):
            for block_index, block in enumerate(self.blocks[stage_index]):
                block_prefix = f"blocks.{stage_index}.{block_index}"
                source_channels = source[f"{block_prefix}.pw_exp.conv.weight"].shape[0]
                target_channels = block.pw_exp.conv.weight.shape[0]
                if target_channels < source_channels:
                    selected_by_block[block_prefix] = self._channel_indices(
                        source, block_prefix, target_channels
                    )

        mapped: dict[str, torch.Tensor] = {}
        for name, target_tensor in target.items():
            if name not in source:
                raise RuntimeError(f"Trained Conv-M checkpoint is missing backbone tensor: {name}")
            source_tensor = source[name]
            if source_tensor.shape == target_tensor.shape:
                mapped[name] = source_tensor
                continue

            block_prefix = ".".join(name.split(".")[:3])
            indices = selected_by_block.get(block_prefix)
            if indices is None:
                raise RuntimeError(
                    f"Unexpected structured-slim mismatch for {name}: "
                    f"source={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
                )
            suffix = name[len(block_prefix) + 1:]
            if suffix == "pw_exp.conv.weight":
                mapped[name] = source_tensor.index_select(0, indices)
            elif suffix.startswith("pw_exp.bn.") or suffix.startswith("dw_mid.bn."):
                mapped[name] = source_tensor if source_tensor.ndim == 0 else source_tensor.index_select(0, indices)
            elif suffix == "dw_mid.conv.weight":
                mapped[name] = source_tensor.index_select(0, indices)
            elif suffix == "pw_proj.conv.weight":
                mapped[name] = source_tensor.index_select(1, indices)
            else:
                raise RuntimeError(f"Unsupported structured-slim tensor: {name}")
            if mapped[name].shape != target_tensor.shape:
                raise RuntimeError(
                    f"Structured-slim produced wrong shape for {name}: "
                    f"mapped={tuple(mapped[name].shape)} target={tuple(target_tensor.shape)}"
                )

        self.load_state_dict(mapped, strict=True)
        return {
            "source_backbone_tensors": len(source),
            "loaded_backbone_tensors": len(mapped),
            "slimmed_blocks": len(selected_by_block),
            "selected_channels": {
                prefix: int(indices.numel()) for prefix, indices in selected_by_block.items()
            },
        }
