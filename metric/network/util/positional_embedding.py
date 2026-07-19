import torch
from math import log2, pi


def generate_fourier_features(
    x: torch.Tensor,
    dim: int = 64,
    max_freq: int = 64,
    use_cos: bool = False,
    use_log: bool = True,
    cat_orig: bool = False,
):
    """
    Minimal copy from UniDepth: projects input coords/angles into Fourier features.
    Args:
        x: (..., input_dim)
        dim: output dimension (must be divisible by input_dim if use_cos=False).
        max_freq: max frequency.
        use_cos: include cosine pairs (default False, only sine).
        use_log: log-spaced frequencies (default True).
        cat_orig: whether to append original x.
    Returns:
        (..., dim) or (..., dim + input_dim) if cat_orig.
    """
    x_orig = x
    device, dtype, input_dim = x.device, x.dtype, x.shape[-1]
    num_bands = dim // (2 * input_dim) if use_cos else dim // input_dim

    if use_log:
        exponents = torch.linspace(
            0.0, log2(max_freq), steps=num_bands, device=device, dtype=dtype
        )
        scales = torch.pow(x.new_tensor(2.0), exponents).to(dtype=dtype)
    else:
        scales = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)

    x = x.unsqueeze(-1)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * x.new_tensor(pi)
    x = torch.cat(
        (
            [x.sin(), x.cos()] if use_cos else [x.sin(),]
        ),
        dim=-1,
    )
    x = x.flatten(-2)
    if cat_orig:
        return torch.cat((x, x_orig), dim=-1)
    return x
