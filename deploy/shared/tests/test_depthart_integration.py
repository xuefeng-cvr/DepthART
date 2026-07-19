import sys
from pathlib import Path

import pytest
import torch

from depthart_selective_scan.cross_scan import cross_selective_scan


DEPTHART_RELATIVE = Path(__file__).resolve().parents[3] / "relative"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cross_scan_matches_depthart_fp32():
    sys.path.insert(0, str(DEPTHART_RELATIVE))
    try:
        from tinyvim.model import tvimblock
    finally:
        sys.path.pop(0)

    torch.manual_seed(23)
    batch, dim, height, width = 1, 16, 7, 9
    directions, rank, state_dim = 4, 2, 4
    x = torch.randn(batch, dim, height, width, device="cuda")
    x_proj = torch.randn(directions, rank + 2 * state_dim, dim, device="cuda") * 0.1
    dt_proj = torch.randn(directions, dim, rank, device="cuda") * 0.1
    dt_bias = torch.randn(directions, dim, device="cuda") * 0.1
    A_logs = torch.randn(directions * dim, state_dim, device="cuda") * 0.1
    Ds = torch.randn(directions * dim, device="cuda") * 0.1
    norm = torch.nn.LayerNorm(dim, device="cuda")
    arguments = (x, x_proj, None, dt_proj, dt_bias, A_logs, Ds, norm)

    expected = tvimblock.cross_selective_scan(*arguments, nrows=1, force_fp32=False)
    actual = cross_selective_scan(*arguments, nrows=1, force_fp32=False)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cross_scan_amp_stays_half():
    torch.manual_seed(29)
    dim, state_dim, rank = 16, 4, 2
    # In a real autocast model this tensor is produced by the preceding conv
    # and is already FP16 when it enters SS2D.
    x = torch.randn(1, dim, 7, 9, device="cuda", dtype=torch.float16)
    args = (
        x,
        torch.randn(4, rank + 2 * state_dim, dim, device="cuda"),
        None,
        torch.randn(4, dim, rank, device="cuda"),
        torch.randn(4, dim, device="cuda"),
        torch.randn(4 * dim, state_dim, device="cuda"),
        torch.randn(4 * dim, device="cuda"),
        torch.nn.LayerNorm(dim, device="cuda"),
    )
    with torch.autocast("cuda", dtype=torch.float16):
        output = cross_selective_scan(*args, nrows=1, force_fp32=False)
    assert output.dtype == torch.float16
