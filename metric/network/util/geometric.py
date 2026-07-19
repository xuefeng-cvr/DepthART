import torch
from torch.nn import functional as F


def generate_rays(camera_intrinsics: torch.Tensor, image_shape: tuple[int, int], noisy: bool = False):
    """
    Minimal copy of UniDepth ray generator.
    Args:
        camera_intrinsics: [B,3,3] intrinsic matrices.
        image_shape: (H, W) of the target feature map.
        noisy: jitter pixel centers (unused for our pipeline).
    Returns:
        ray_directions: [B, H*W, 3] unit vectors per pixel.
        angles: [B, H*W, 2] (theta, phi) spherical angles (kept for parity).
    """
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape

    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    zeros = torch.zeros(batch_size, device=device, dtype=dtype)
    ones = torch.ones(batch_size, device=device, dtype=dtype)
    intrinsics_inv = torch.stack(
        (
            torch.stack((1.0 / camera_intrinsics[:, 0, 0], zeros, -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]), dim=-1),
            torch.stack((zeros, 1.0 / camera_intrinsics[:, 1, 1], -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=1,
    )

    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (B, 3, H*W)
    ray_directions = F.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles
