"""Inference-only DepthART models with MobileNetV4 encoders."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tinyvim.model.util.blocks import FeatureFusionBlock, _make_scratch

from .backbone import MobileNetV4ConvFeatures, MobileNetV4MSlimFeatures


def _make_fusion_block(features: int, use_bn: bool) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPTHead(nn.Module):
    """Original DepthART DPT head adapted to MobileNetV4 feature widths."""

    def __init__(self, features: int, out_channels: list[int], use_bn: bool = False):
        super().__init__()
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        self.scratch.output_conv1 = nn.Conv2d(
            features, features // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(
        self, features: list[torch.Tensor], size_h: int, size_w: int
    ) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1 = self.scratch.layer1_rn(layer_1)
        layer_2 = self.scratch.layer2_rn(layer_2)
        layer_3 = self.scratch.layer3_rn(layer_3)
        layer_4 = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4, size=layer_3.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3, size=layer_2.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1)
        output = self.scratch.output_conv1(path_1)
        output = F.interpolate(
            output, (size_h, size_w), mode="bilinear", align_corners=True
        )
        return self.scratch.output_conv2(output)


class MobileNetV4Depth(nn.Module):
    """MobileNetV4-Conv-S/M encoder with the original DepthART DPT head."""

    def __init__(self, encoder: str = "S", use_bn: bool = False):
        super().__init__()
        self.pretrained = MobileNetV4ConvFeatures(encoder)
        self.encoder_variant = self.pretrained.variant
        decoder_features = 48 if self.encoder_variant == "small" else 64
        self.depth_head = DPTHead(
            features=decoder_features,
            out_channels=self.pretrained.out_channels,
            use_bn=use_bn,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pretrained(inputs)
        return self.depth_head(features, inputs.shape[-2], inputs.shape[-1])


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SPFResidualRefiner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels, kernel_size=3)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(inputs + self.bn2(self.conv2(self.conv1(inputs))))


class SPFDepthwiseRefiner(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.depthwise = ConvBNReLU(
            channels, channels, kernel_size=kernel_size, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.pointwise_bn(self.pointwise(self.depthwise(inputs)))
        return self.relu(inputs + outputs)


class SinglePathPyramidFusionDecoder(nn.Module):
    """Low-resolution-first decoder used by MobileNetV4-M-slim-SPF."""

    def __init__(self):
        super().__init__()
        self.project8 = ConvBNReLU(80, 96)
        self.project16 = ConvBNReLU(160, 96)
        self.project32 = ConvBNReLU(256, 96)
        self.fuse8 = ConvBNReLU(96 * 3, 96)
        self.refine8 = SPFResidualRefiner(96)
        self.reduce8 = ConvBNReLU(96, 64)
        self.project4 = ConvBNReLU(48, 64)
        self.refine4 = SPFDepthwiseRefiner(64, kernel_size=5)
        self.reduce4 = ConvBNReLU(64, 32)
        self.project2 = ConvBNReLU(32, 32)
        self.refine2 = SPFDepthwiseRefiner(32, kernel_size=3)
        self.reduce2 = ConvBNReLU(32, 16)
        self.full_depthwise = ConvBNReLU(16, 16, kernel_size=3, groups=16)
        self.full_pointwise = ConvBNReLU(16, 16)
        self.output_conv = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.output_relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _resize(inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(inputs, size=size, mode="bilinear", align_corners=True)

    def forward(
        self,
        stem2: torch.Tensor,
        features: list[torch.Tensor],
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        feature4, feature8, feature16, feature32 = features
        size8 = feature8.shape[-2:]
        fused8 = torch.cat(
            (
                self.project8(feature8),
                self._resize(self.project16(feature16), size8),
                self._resize(self.project32(feature32), size8),
            ),
            dim=1,
        )
        fused8 = self.refine8(self.fuse8(fused8))
        fused4 = self._resize(self.reduce8(fused8), feature4.shape[-2:])
        fused4 = self.refine4(fused4 + self.project4(feature4))
        fused2 = self._resize(self.reduce4(fused4), stem2.shape[-2:])
        fused2 = self.refine2(fused2 + self.project2(stem2))
        full = self._resize(self.reduce2(fused2), (output_height, output_width))
        full = self.full_pointwise(self.full_depthwise(full))
        return self.output_relu(self.output_conv(full))


class MobileNetV4MSlimSPFDepth(nn.Module):
    """MobileNetV4-Conv-M with slim expansion widths and the SPF decoder."""

    def __init__(self):
        super().__init__()
        self.pretrained = MobileNetV4MSlimFeatures()
        self.encoder_variant = self.pretrained.variant
        self.depth_head = SinglePathPyramidFusionDecoder()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stem2, features = self.pretrained.forward_with_stem(inputs)
        return self.depth_head(stem2, features, inputs.shape[-2], inputs.shape[-1])
