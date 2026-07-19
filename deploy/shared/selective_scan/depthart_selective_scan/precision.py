from contextlib import contextmanager, nullcontext

import torch


def tf32_available(device=None):
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


@contextmanager
def precision_context(mode="fp32", device_type="cuda"):
    """Configure model-level precision around Selective Scan.

    Selective Scan has no TF32 storage type. In tf32 mode its recurrence remains
    FP32 while surrounding GEMM/conv operations may use TF32 on Ampere or newer.
    """
    mode = mode.lower()
    if mode not in ("fp32", "tf32", "fp16", "amp"):
        raise ValueError(f"unsupported precision mode: {mode}")
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    enable_tf32 = mode == "tf32" and tf32_available()
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    if mode == "amp" and hasattr(torch, "autocast"):
        autocast = torch.autocast(device_type, dtype=torch.float16)
    elif mode == "amp" and device_type == "cuda":
        autocast = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        autocast = nullcontext()
    try:
        with autocast:
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
