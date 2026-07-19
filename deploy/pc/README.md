# PC Deployment

This guide covers Linux workstations and servers with an NVIDIA CUDA GPU.

## Environment

Create the tested PyTorch environment from the repository root:

```bash
conda env create -f environment.yml
conda activate depthart
```

The configuration pins Python 3.9, PyTorch 2.1, CUDA 11.8, and all Python dependencies required for inference, ONNX validation, and extension compilation.

Check the CUDA toolchain:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
nvcc --version
```

## PyTorch Setup

Compile the Selective Scan extension for the current GPU:

```bash
MAX_JOBS=4 bash deploy/pc/setup.sh pytorch-only
```

After compilation, the standard `relative/infer_image.py` and `metric/infer_image.py` commands automatically use the optimized CUDA operator. If the extension is unavailable, they report `reference fallback` and remain functional but are not suitable for speed measurements.

## TensorRT Setup

The custom TensorRT plugin requires both Python bindings and C++ development files:

- `NvInfer.h`;
- `libnvinfer`;
- `libnvinfer_plugin`.

Install TensorRT using NVIDIA's Debian packages, tar package, or development container. A pip-only TensorRT install does not include the C++ headers required to compile this plugin.

Verify TensorRT, then compile both backends:

```bash
python -c 'import tensorrt as trt; print(trt.__version__)'
MAX_JOBS=4 bash deploy/pc/setup.sh
```

For a nonstandard TensorRT installation:

```bash
export TENSORRT_INCLUDE_DIR=/path/to/include
export TENSORRT_LIB_DIR=/path/to/lib
export TENSORRT_NVINFER_LIBRARY=/path/to/libnvinfer.so
export TENSORRT_PLUGIN_LIBRARY=/path/to/libnvinfer_plugin.so
MAX_JOBS=4 bash deploy/pc/setup.sh
```

## Benchmark

```bash
bash deploy/pc/benchmark.sh \
  --stage all \
  --samples 1000 \
  --warmup 200 \
  --workspace-gb 8
```

Results are written to `deploy/pc/runs/local` by default. Set a different output directory with:

```bash
DEPTHART_OUTPUT_ROOT=/path/to/run bash deploy/pc/benchmark.sh --stage all
```

The formal RTX A6000 reference results are available in [`results/a6000`](results/a6000).
