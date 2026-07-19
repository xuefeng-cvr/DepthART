import warnings

import torch

from .reference import selective_scan_reference


try:
    import depthart_selective_scan_cuda as _C
except ImportError as error:
    _C = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


def _normalize_inputs(u, delta, A, B, C, D, delta_bias):
    if B.ndim == 3:
        B = B.unsqueeze(1)
    if C.ndim == 3:
        C = C.unsqueeze(1)
    u = u.contiguous() if u.stride(-1) != 1 else u
    delta = delta.contiguous() if delta.stride(-1) != 1 else delta
    B = B.contiguous() if B.stride(-1) != 1 else B
    C = C.contiguous() if C.stride(-1) != 1 else C
    A = A.float().contiguous()
    D = None if D is None else D.float().contiguous()
    delta_bias = None if delta_bias is None else delta_bias.float().contiguous()
    return u, delta, A, B, C, D, delta_bias


def _extension_forward(u, delta, A, B, C, D, delta_bias, delta_softplus, out_float):
    if _C is None:
        raise RuntimeError(f"CUDA extension is not installed: {_IMPORT_ERROR}")
    tensors = _normalize_inputs(u, delta, A, B, C, D, delta_bias)
    output, state = _C.fwd(*tensors, delta_softplus, 1, out_float)
    return output, state, tensors


class _SelectiveScanAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias, delta_softplus, out_float):
        ctx.squeeze_B = B.ndim == 3
        ctx.squeeze_C = C.ndim == 3
        output, state, tensors = _extension_forward(
            u, delta, A, B, C, D, delta_bias, delta_softplus, out_float
        )
        u, delta, A, B, C, D, delta_bias = tensors
        ctx.has_D = D is not None
        ctx.has_delta_bias = delta_bias is not None
        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(
            u,
            delta,
            A,
            B,
            C,
            D if D is not None else u.new_empty(0, dtype=torch.float32),
            delta_bias if delta_bias is not None else u.new_empty(0, dtype=torch.float32),
            state,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        u, delta, A, B, C, D_saved, bias_saved, state = ctx.saved_tensors
        D = D_saved if ctx.has_D else None
        delta_bias = bias_saved if ctx.has_delta_bias else None
        gradients = _C.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            grad_output.contiguous(),
            state,
            ctx.delta_softplus,
            1,
        )
        du, ddelta, dA, dB, dC, dD, dbias = gradients
        if ctx.squeeze_B:
            dB = dB.squeeze(1)
        if ctx.squeeze_C:
            dC = dC.squeeze(1)
        return du, ddelta, dA, dB, dC, dD if ctx.has_D else None, dbias if ctx.has_delta_bias else None, None, None


def _cuda_impl(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, out_float=False):
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in (u, delta, A, B, C, D, delta_bias)
    ):
        return _SelectiveScanAutograd.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, out_float)
    return _extension_forward(u, delta, A, B, C, D, delta_bias, delta_softplus, out_float)[0]


def _cpu_impl(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, out_float=False):
    return selective_scan_reference(u, delta, A, B, C, D, delta_bias, delta_softplus, out_float)


def _meta_impl(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, out_float=False):
    dtype = torch.float32 if out_float else u.dtype
    return torch.empty_like(u, dtype=dtype, device="meta")


def _autocast_impl(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, out_float=False):
    get_dtype = getattr(torch, "get_autocast_gpu_dtype", None)
    dtype = get_dtype() if get_dtype is not None else torch.float16
    with torch.cuda.amp.autocast(enabled=False):
        return torch.ops.depthart.selective_scan(
            u.to(dtype),
            delta.to(dtype),
            A.float(),
            B.to(dtype),
            C.to(dtype),
            None if D is None else D.float(),
            None if delta_bias is None else delta_bias.float(),
            delta_softplus,
            out_float,
        )


_LIBRARIES = []


def _register_ops():
    try:
        definition = torch.library.Library("depthart", "DEF")
        definition.define(
            "selective_scan(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, "
            "Tensor? D=None, Tensor? delta_bias=None, bool delta_softplus=True, "
            "bool out_float=False) -> Tensor"
        )
        cuda = torch.library.Library("depthart", "IMPL", "CUDA")
        cpu = torch.library.Library("depthart", "IMPL", "CPU")
        meta = torch.library.Library("depthart", "IMPL", "Meta")
        autocast = torch.library.Library("depthart", "IMPL", "AutocastCUDA")
        cuda.impl("selective_scan", _cuda_impl)
        cpu.impl("selective_scan", _cpu_impl)
        meta.impl("selective_scan", _meta_impl)
        autocast.impl("selective_scan", _autocast_impl)
        _LIBRARIES.extend((definition, cuda, cpu, meta, autocast))
        return True
    except (AttributeError, RuntimeError) as error:
        warnings.warn(f"torch.library registration unavailable; using eager fallback: {error}")
        return False


_REGISTERED = _register_ops()


def selective_scan(
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
    """Run Selective Scan with AMP-aware mixed precision.

    Dynamic inputs may be FP16 or FP32. Recurrence weights A, D and
    delta_bias are always normalized to FP32 by the CUDA implementation.
    """
    if _REGISTERED:
        return torch.ops.depthart.selective_scan(
            u, delta, A, B, C, D, delta_bias, delta_softplus, out_float
        )
    if u.is_cuda:
        return _cuda_impl(u, delta, A, B, C, D, delta_bias, delta_softplus, out_float)
    return _cpu_impl(u, delta, A, B, C, D, delta_bias, delta_softplus, out_float)
