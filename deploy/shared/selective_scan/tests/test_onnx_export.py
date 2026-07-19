from pathlib import Path

import pytest
import torch

from depthart_selective_scan import register_onnx_symbolic, selective_scan


class ScanModule(torch.nn.Module):
    def forward(self, u, delta, A, B, C, D, bias):
        return selective_scan(u, delta, A, B, C, D, bias)


def test_onnx_custom_node(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    register_onnx_symbolic(17)
    batch, dim, length, groups, state = 1, 8, 16, 2, 4
    tensors = (
        torch.randn(batch, dim, length),
        torch.randn(batch, dim, length),
        -torch.rand(dim, state),
        torch.randn(batch, groups, state, length),
        torch.randn(batch, groups, state, length),
        torch.randn(dim),
        torch.randn(dim),
    )
    output = tmp_path / "selective_scan.onnx"
    torch.onnx.export(
        ScanModule(),
        tensors,
        output,
        opset_version=17,
        input_names=("u", "delta", "A", "B", "C", "D", "delta_bias"),
        output_names=("output",),
    )
    graph = onnx.load(output).graph
    nodes = [node for node in graph.node if node.domain == "com.depthart"]
    assert len(nodes) == 1
    assert nodes[0].op_type == "SelectiveScan"
    assert len(nodes[0].input) == 7
