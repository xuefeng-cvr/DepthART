# DepthART Selective Scan

Parameter-preserving PyTorch CUDA operator and TensorRT plugin for the Selective Scan recurrence used by DepthART.

## Precision

| Mode | Dynamic tensors | Recurrence parameters/accumulation |
| --- | --- | --- |
| FP32 | FP32 | FP32 |
| TF32 | FP32 | FP32; TF32 applies to surrounding GEMM/conv only |
| FP16 | FP16 | FP32 |
| AMP | autocast FP16 | FP32 |

The scan is a recurrence rather than a matrix multiplication, so TF32 does not directly accelerate the scan kernel. FP32 recurrence arithmetic is retained for numerical stability.

For Relative-L-224 TensorRT FP16, the engine builder also keeps the small final depth prediction head in FP32. The released checkpoint can exceed the FP16 numeric range in that head on real images; `run_speed_matrix.py` and `run_backend_evaluation.py` enable this policy automatically.

## Build targets

From `deploy/shared/selective_scan`:

```bash
# PyTorch extension
bash scripts/build_for_device.sh orin-nx

# TensorRT plugin
bash scripts/build_trt_plugin.sh orin-nx

# Tests
python -m pytest -q tests
```

Supported target labels are `native`, `orin-nx` (SM87), `xavier-nx` (SM72), and `jetson-nano` (SM53). Build on the target device: generated binaries depend on architecture, CUDA, PyTorch, TensorRT, Python, and the platform ABI.

The TensorRT build writes a stable local plugin path:

```text
results/plugins/libdepthart_selective_scan_trt.so
```

Standard Jetson TensorRT header/library paths are detected automatically. Nonstandard installations can be supplied with:

```bash
export TENSORRT_INCLUDE_DIR=/path/to/include
export TENSORRT_LIB_DIR=/path/to/lib
export TENSORRT_NVINFER_LIBRARY=/path/to/libnvinfer.so
export TENSORRT_PLUGIN_LIBRARY=/path/to/libnvinfer_plugin.so
```

## Parameter invariance

`optimize_depthart_inference` only skips Relative auxiliary tensors that the released head discards. Metric deployment caches `DenseCameraEmbedder` and DAA K/V outputs for fixed intrinsics. Both paths hash the full state dict before and after preparation and fail if any parameter changes.

## ONNX contract

The static export uses `com.depthart::SelectiveScan` with inputs:

```text
u, delta, A, B, C, D, delta_bias
```

Attributes are `delta_softplus` and `out_float`. TensorRT requires `trt_plugin/selective_scan_plugin.cpp`; the PyTorch extension is not a TensorRT plugin.

This module owns the custom-node contract only. Complete exported DepthART model
graphs are stored separately in `deploy/shared/onnx/`.
