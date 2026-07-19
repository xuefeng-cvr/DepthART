import pytest
import torch

from depthart_selective_scan import selective_scan, selective_scan_reference


def make_inputs(device, dtype=torch.float32, requires_grad=False):
    torch.manual_seed(17)
    batch, dim, length, groups, state_dim = 2, 8, 19, 2, 4
    dynamic = lambda *shape: torch.randn(*shape, device=device, dtype=dtype) * 0.1
    u = dynamic(batch, dim, length).requires_grad_(requires_grad)
    delta = dynamic(batch, dim, length).requires_grad_(requires_grad)
    A = (-torch.rand(dim, state_dim, device=device)).requires_grad_(requires_grad)
    B = dynamic(batch, groups, state_dim, length).requires_grad_(requires_grad)
    C = dynamic(batch, groups, state_dim, length).requires_grad_(requires_grad)
    D = torch.randn(dim, device=device).requires_grad_(requires_grad)
    bias = torch.randn(dim, device=device).requires_grad_(requires_grad)
    return u, delta, A, B, C, D, bias


def test_cpu_dispatch_matches_reference():
    inputs = make_inputs("cpu")
    actual = selective_scan(*inputs)
    expected = selective_scan_reference(*inputs)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cuda_matches_reference(dtype):
    inputs = make_inputs("cuda", dtype)
    actual = selective_scan(*inputs)
    expected = selective_scan_reference(*inputs)
    assert actual.dtype == dtype
    tolerance = dict(rtol=2e-4, atol=2e-5) if dtype == torch.float32 else dict(rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(actual.float(), expected.float(), **tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_float_output_from_half_input():
    inputs = make_inputs("cuda", torch.float16)
    output = selective_scan(*inputs, out_float=True)
    assert output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_amp_casts_dynamic_inputs_to_half():
    inputs = make_inputs("cuda", torch.float32)
    with torch.autocast("cuda", dtype=torch.float16):
        output = selective_scan(*inputs)
    assert output.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backward_is_finite():
    inputs = make_inputs("cuda", torch.float32, requires_grad=True)
    selective_scan(*inputs).square().mean().backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backward_matches_reference():
    custom_inputs = make_inputs("cuda", torch.float32, requires_grad=True)
    reference_inputs = tuple(value.detach().clone().requires_grad_(True) for value in custom_inputs)
    torch.manual_seed(31)
    gradient = torch.randn_like(custom_inputs[0])
    selective_scan(*custom_inputs).backward(gradient)
    selective_scan_reference(*reference_inputs).backward(gradient)
    for actual, expected in zip(custom_inputs, reference_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-3, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backward_supports_ungrouped_3d_BC():
    inputs = list(make_inputs("cuda", torch.float32, requires_grad=True))
    inputs[3] = inputs[3][:, 0].detach().clone().requires_grad_(True)
    inputs[4] = inputs[4][:, 0].detach().clone().requires_grad_(True)
    selective_scan(*inputs).mean().backward()
    assert inputs[3].grad.shape == inputs[3].shape
    assert inputs[4].grad.shape == inputs[4].shape


def test_meta_dispatch_shape_and_dtype():
    inputs = make_inputs("meta")
    output = selective_scan(*inputs)
    assert output.device.type == "meta"
    assert output.shape == inputs[0].shape
    assert output.dtype == inputs[0].dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_torch_compile_fullgraph():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    inputs = make_inputs("cuda")
    compiled = torch.compile(selective_scan, fullgraph=True)
    actual = compiled(*inputs)
    expected = selective_scan(*inputs)
    torch.testing.assert_close(actual, expected)
