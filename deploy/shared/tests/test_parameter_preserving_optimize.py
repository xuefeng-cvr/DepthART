import sys
from pathlib import Path

import pytest
import torch

from depthart_selective_scan import (
    CUDAGraphRunner,
    disable_unused_auxiliary_features,
    parameter_fingerprint,
    restore_auxiliary_forward,
)


ROOT = Path(__file__).resolve().parents[3]
DEPTHART_RELATIVE = ROOT / "relative"
CHECKPOINT = ROOT / "checkpoints/relative/depthart_relative_s_448.pth"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_disable_auxiliary_is_parameter_and_output_invariant():
    sys.path.insert(0, str(DEPTHART_RELATIVE))
    try:
        from tinyvim.model.dpt import TinyVimDepth
    finally:
        sys.path.pop(0)
    model = TinyVimDepth(encoder="S").cuda().eval()
    payload = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(payload.get("model", payload), strict=True)
    image = torch.randn(1, 3, 224, 224, device="cuda")
    before = parameter_fingerprint(model)
    with torch.inference_mode():
        expected = model(image)
        disable_unused_auxiliary_features(model)
        actual = model(image)
    assert parameter_fingerprint(model) == before
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restore_auxiliary_forward(model)
    with torch.inference_mode():
        restored = model(image)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_updates_for_different_inputs():
    sys.path.insert(0, str(DEPTHART_RELATIVE))
    try:
        from tinyvim.model.dpt import TinyVimDepth
        from tinyvim.model import tvimblock
        from depthart_selective_scan import install_depthart
    finally:
        sys.path.pop(0)
    model = TinyVimDepth(encoder="S").cuda().eval()
    payload = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(payload.get("model", payload), strict=True)
    disable_unused_auxiliary_features(model)
    install_depthart(tvimblock)
    first = torch.randn(1, 3, 224, 224, device="cuda")
    second = torch.randn_like(first)
    runner = CUDAGraphRunner(model, first, amp=False, warmup=5)
    with torch.inference_mode():
        expected_first = model(first)
        expected_second = model(second)
        actual_first = runner(first).clone()
        actual_second = runner(second).clone()
    torch.testing.assert_close(actual_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)
    assert not torch.equal(actual_first, actual_second)
