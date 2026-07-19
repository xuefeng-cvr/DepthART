import torch
import torch.nn.functional as F


def selective_scan_reference(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    delta_bias=None,
    delta_softplus=True,
    out_float=False,
):
    """Portable reference implementation used for CPU tests and validation."""
    input_dtype = u.dtype
    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    if B_f.ndim == 3:
        B_f = B_f.unsqueeze(1)
    if C_f.ndim == 3:
        C_f = C_f.unsqueeze(1)
    if delta_bias is not None:
        delta_f = delta_f + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta_f = F.softplus(delta_f)

    batch, dim, length = u_f.shape
    groups = B_f.shape[1]
    state_dim = B_f.shape[2]
    group_size = dim // groups
    delta_repeats = dim // delta_f.shape[1]
    if delta_repeats != 1:
        delta_f = delta_f.repeat_interleave(delta_repeats, dim=1)
    B_f = B_f.repeat_interleave(group_size, dim=1)
    C_f = C_f.repeat_interleave(group_size, dim=1)
    state = u_f.new_zeros(batch, dim, state_dim)
    outputs = []
    for index in range(length):
        dt = delta_f[:, :, index]
        transition = torch.exp(dt.unsqueeze(-1) * A_f.unsqueeze(0))
        state = transition * state + dt.unsqueeze(-1) * B_f[:, :, :, index] * u_f[:, :, index].unsqueeze(-1)
        value = (state * C_f[:, :, :, index]).sum(-1)
        if D is not None:
            value = value + D.float().view(1, -1) * u_f[:, :, index]
        outputs.append(value)
    output = torch.stack(outputs, dim=-1)
    return output if out_float else output.to(input_dtype)
