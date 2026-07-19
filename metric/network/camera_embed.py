import torch
import torch.nn as nn

from .util.geometric import generate_rays
from .util.positional_embedding import generate_fourier_features


class DenseCameraEmbedder(nn.Module):
    """
    Generate dense camera embeddings using Fourier features on polar/azimuth angles (UniDepth v2 style).
    K is assumed fixed; we scale fx, fy, cx, cy by the stage downsample ratio.
    """

    def __init__(self, intrinsic: torch.Tensor, cam_dims=(256, 256, 256, 256)):
        super().__init__()
        self.register_buffer("K0", intrinsic.clone())
        self.cache = {}
        self.cam_dims = cam_dims

    def _scale_K(self, scale: int):
        K = self.K0.clone()
        K[0, 0] /= scale
        K[1, 1] /= scale
        K[0, 2] /= scale
        K[1, 2] /= scale
        return K

    def _embed(self, H: int, W: int, scale: int, dim: int, device):
        key = (H, W, scale, dim)
        if key in self.cache:
            return self.cache[key].to(device)
        Ks = self._scale_K(scale).to(device)
        rays, _ = generate_rays(Ks[None], (H // scale, W // scale))
        rays = rays.view(1, H // scale, W // scale, 3)
        rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        polar = torch.acos(z)
        x_clipped = x.abs().clamp(min=1e-3) * (2 * (x >= 0).int() - 1)
        azimuth = torch.atan2(y, x_clipped)
        angles = torch.stack([polar, azimuth], dim=-1)  # (1, Hs, Ws, 2)
        emb = generate_fourier_features(
            angles,
            dim=dim,
            max_freq=max(H // scale, W // scale) // 2,
            use_log=True,
            cat_orig=False,
        )  # (1, Hs, Ws, dim)
        emb = emb.permute(0, 3, 1, 2).contiguous()  # (1, dim, Hs, Ws)
        self.cache[key] = emb.detach()
        return emb

    def forward(self, H: int, W: int, device=None):
        device = device or self.K0.device
        return (
            self._embed(H, W, 4, self.cam_dims[0], device),
            self._embed(H, W, 8, self.cam_dims[1], device),
            self._embed(H, W, 16, self.cam_dims[2], device),
            self._embed(H, W, 32, self.cam_dims[3], device),
        )
