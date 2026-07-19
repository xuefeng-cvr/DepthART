import torch

from .ops import selective_scan


def cross_selective_scan_2d(
    x,
    x_proj_weight,
    x_proj_bias,
    dt_projs_weight,
    dt_projs_bias,
    A_logs,
    Ds,
    out_norm,
    nrows=-1,
    delta_softplus=True,
    to_dtype=True,
    force_fp32=False,
):
    """DepthART-compatible four-direction SS2D composition.

    ``nrows`` is accepted for source compatibility. The portable CUDA backend
    uses its nrows=1 kernel, which is also what DepthART's SS2D currently asks
    for in ``forward_core``.
    """
    del nrows
    input_dtype = x.dtype
    batch, channels, height, width = x.shape
    length = height * width
    directions, _, rank = dt_projs_weight.shape
    state_dim = A_logs.shape[1]
    horizontal = x.flatten(2)
    vertical = x.transpose(2, 3).flatten(2)
    scans = torch.stack((horizontal, vertical, horizontal.flip(-1), vertical.flip(-1)), dim=1)
    projected = torch.einsum("b k d l, k c d -> b k c l", scans, x_proj_weight)
    if x_proj_bias is not None:
        projected = projected + x_proj_bias.view(1, directions, -1, 1)
    dt, B, C = torch.split(projected, (rank, state_dim, state_dim), dim=2)
    dt = torch.einsum("b k r l, k d r -> b k d l", dt, dt_projs_weight)
    dynamic = (scans.reshape(batch, -1, length), dt.contiguous().reshape(batch, -1, length), B.contiguous(), C.contiguous())
    if force_fp32:
        dynamic = tuple(value.float() for value in dynamic)
    scans_flat, dt_flat, B, C = dynamic
    output = selective_scan(
        scans_flat,
        dt_flat,
        -torch.exp(A_logs.float()),
        B,
        C,
        Ds.float(),
        dt_projs_bias.reshape(-1).float(),
        delta_softplus,
        False,
    ).view(batch, directions, -1, length)
    merged = output[:, 0:2] + output[:, 2:4].flip(-1)
    y = merged[:, 0] + merged[:, 1].view(batch, -1, width, height).transpose(2, 3).reshape(batch, -1, length)
    y = y.transpose(1, 2).contiguous()
    if out_norm is not None:
        y = out_norm(y)
    y = y.view(batch, height, width, channels)
    return y.to(input_dtype) if to_dtype else y


# Match the function name used by DepthART's tvimblock.py.
cross_selective_scan = cross_selective_scan_2d
