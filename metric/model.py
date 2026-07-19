"""K-aware metric DepthART model used by the released indoor/outdoor weights."""

from pathlib import Path

import torch
import torch.nn as nn
from timm.models import create_model

from network import tinyvim as _tinyvim_registration  # noqa: F401
from network.daa import DAAStage
from network.dpt import DPTHead
from network.sfh import ScaleFormerHead
from network.util.geometric import generate_rays
from network.util.positional_embedding import generate_fourier_features


ENCODER_SPECS = {
    "S": (48, (48, 64, 168, 224)),
    "B": (48, (48, 96, 192, 384)),
    "L": (48, (64, 128, 384, 512)),
}


class DenseCameraEmbedder(nn.Module):
    def __init__(self, cam_dims=(256, 256, 256, 256)):
        super().__init__()
        self.cam_dims = cam_dims

    @staticmethod
    def _embed(K, height, width, scale, dim, device):
        K = K.to(device=device, dtype=torch.float32)
        Ks = torch.stack(
            (
                torch.stack((K[:, 0, 0] / scale, K[:, 0, 1] / scale, K[:, 0, 2] / scale), dim=-1),
                torch.stack((K[:, 1, 0] / scale, K[:, 1, 1] / scale, K[:, 1, 2] / scale), dim=-1),
                K[:, 2, :],
            ),
            dim=1,
        )
        rays, _ = generate_rays(Ks, (height // scale, width // scale))
        rays = rays.view(Ks.shape[0], height // scale, width // scale, 3)
        rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x, y, z = rays.unbind(dim=-1)
        polar = torch.acos(z.clamp(-1.0, 1.0))
        signed_x = x.abs().clamp(min=1e-3) * (2 * (x >= 0).int() - 1)
        if torch.onnx.is_in_onnx_export():
            base = torch.atan(y / signed_x)
            correction = torch.where(y >= 0, base.new_tensor(torch.pi), base.new_tensor(-torch.pi))
            azimuth = torch.where(signed_x < 0, base + correction, base)
        else:
            azimuth = torch.atan2(y, signed_x)
        angles = torch.stack((polar, azimuth), dim=-1)
        embedding = generate_fourier_features(
            angles,
            dim=dim,
            max_freq=max(height // scale, width // scale) // 2,
            use_log=True,
            cat_orig=False,
        )
        return embedding.permute(0, 3, 1, 2).contiguous()

    def forward(self, K, height, width, device=None):
        device = device or K.device
        return tuple(
            self._embed(K, height, width, scale, dim, device)
            for scale, dim in zip((4, 8, 16, 32), self.cam_dims)
        )


class MetricDepthART(nn.Module):
    def __init__(self, encoder="S", max_depth=10.0, sfh_num_queries=8):
        super().__init__()
        encoder = encoder.upper()
        align_channels, out_channels = ENCODER_SPECS[encoder]
        self.pretrained = create_model(
            f"TinyViM_{encoder}", num_classes=1000, pretrained=False, fork_feat=True
        )
        self.depth_head = DPTHead(
            align_channels=align_channels, out_channels=list(out_channels), use_bn=False
        )
        self.cam_embedder = DenseCameraEmbedder()
        self.daa1 = DAAStage(out_channels[0], 256, fusion_method="cross_attention")
        self.daa2 = DAAStage(out_channels[1], 256, fusion_method="cross_attention")
        self.daa3 = DAAStage(out_channels[2], 256, fusion_method="cross_attention")
        self.daa4 = DAAStage(out_channels[3], 256, fusion_method="cross_attention")
        self.sfh = ScaleFormerHead(
            in_dim=out_channels[3], cam_dim=256, num_queries=sfh_num_queries
        )
        self.max_depth = float(max_depth)

    def forward(self, image, K):
        cameras = self.cam_embedder(K, image.shape[-2], image.shape[-1], image.device)
        features = self.pretrained.forward_with_adapters(
            image,
            adapters=[self.daa1, self.daa2, self.daa3, self.daa4],
            cams=list(cameras),
        )
        depth = self.depth_head(features, image.shape[-2], image.shape[-1])
        scale = self.sfh(features[3], cameras[3])
        return (depth * scale.view(-1, 1, 1, 1) * self.max_depth).squeeze(1)


def load_model(checkpoint, encoder, domain, device="cuda"):
    """Build and strictly load one of the released metric checkpoints."""
    max_depth = 10.0 if domain == "indoor" else 80.0
    model = MetricDepthART(encoder=encoder, max_depth=max_depth)
    payload = torch.load(Path(checkpoint), map_location="cpu")
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
