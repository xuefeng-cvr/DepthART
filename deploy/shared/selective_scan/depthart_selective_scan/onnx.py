import torch
from torch.onnx import symbolic_helper


DOMAIN = "com.depthart"
OP_NAME = "SelectiveScan"


def _symbolic(g, u, delta, A, B, C, D, delta_bias, delta_softplus, out_float):
    softplus = symbolic_helper._parse_arg(delta_softplus, "b")
    float_output = symbolic_helper._parse_arg(out_float, "b")
    inputs = [u, delta, A, B, C]
    if D is not None:
        inputs.append(D)
    if delta_bias is not None:
        inputs.append(delta_bias)
    return g.op(
        f"{DOMAIN}::{OP_NAME}",
        *inputs,
        delta_softplus_i=int(softplus),
        out_float_i=int(float_output),
    )


def register_onnx_symbolic(opset=17):
    """Register the ONNX custom-node contract used by deployment plugins."""
    torch.onnx.register_custom_op_symbolic("depthart::selective_scan", _symbolic, opset)
    return f"{DOMAIN}::{OP_NAME}"
