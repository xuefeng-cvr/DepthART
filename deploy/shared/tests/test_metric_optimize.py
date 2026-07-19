import sys
from pathlib import Path

import pytest
import torch

from depthart_selective_scan import (
    CUDAGraphRunner,
    cache_metric_camera_embeddings,
    cache_metric_daa_attention_kv,
    install_depthart,
    parameter_fingerprint,
)


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "checkpoints/metric/depthart_metric_indoor_s_448.pth"


def fixed_K(device):
    return torch.tensor(
        [[[350.0, 0.0, 223.65], [0.0, 466.6666667, 223.5333333], [0.0, 0.0, 1.0]]],
        device=device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_metric_cache_and_graph_are_parameter_and_output_invariant():
    sys.path.insert(0, str(ROOT / "metric"))
    try:
        from model import load_model
        from network import tvimblock
    finally:
        sys.path.pop(0)
    model = load_model(CHECKPOINT, "S", "indoor", "cuda")
    image = torch.randn(1, 3, 448, 448, device="cuda")
    K = fixed_K("cuda")
    fingerprint = parameter_fingerprint(model)
    with torch.inference_mode():
        expected = model(image, K)
        cache_metric_camera_embeddings(model, K, 448, 448)
        cached = model(image)
    assert parameter_fingerprint(model) == fingerprint
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)

    with torch.inference_mode():
        cache_metric_daa_attention_kv(model)
        cached_kv = model(image)
    assert parameter_fingerprint(model) == fingerprint
    torch.testing.assert_close(cached_kv, expected, rtol=0, atol=0)

    install_depthart(tvimblock)
    with torch.inference_mode():
        optimized = model(image)
    torch.testing.assert_close(optimized, expected, rtol=5e-4, atol=1e-4)

    runner = CUDAGraphRunner(model, image, amp=False, warmup=5)
    with torch.inference_mode():
        graphed = runner(image).clone()
    torch.testing.assert_close(graphed, optimized, rtol=0, atol=0)
