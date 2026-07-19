# Jetson Orin NX Deployment

This guide runs DepthART on Jetson Orin NX with PyTorch and TensorRT. All CUDA extensions, plugins, timing caches, and TensorRT engines are built directly on the Orin NX.

## Tested Configuration

| Component | Version |
| --- | --- |
| JetPack | 6.2 |
| Container | `nvcr.io/nvidia/pytorch:25.06-py3` |
| Ubuntu | 24.04 |
| Python | 3.12 |
| PyTorch | `2.8.0a0+5228986c39` |
| CUDA | 12.9.1 |
| TensorRT | 10.11.0.33 |
| GPU architecture | SM87 |

The container definition is available at [`Dockerfile`](Dockerfile). Python dependencies are pinned in [`../shared/requirements.txt`](../shared/requirements.txt).

## 1. Prepare the Orin NX Host

Install JetPack 6.2 and the NVIDIA Container Runtime. Confirm that Docker can access the GPU:

```bash
docker run --rm --runtime nvidia nvcr.io/nvidia/pytorch:25.06-py3 \
  python3 -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
```

Clone the repository with Git LFS:

```bash
git lfs install
git lfs pull
sha256sum -c CHECKSUMS.sha256
```

## 2. Build the Container

```bash
docker build -f deploy/orin_nx/Dockerfile -t depthart:orin-nx .
```

Set the device to maximum-performance mode on the host:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Start the container with the repository mounted read/write:

```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v "$PWD":/workspace/DepthART \
  -w /workspace/DepthART \
  depthart:orin-nx
```

## 3. Compile Selective Scan

Inside the container:

```bash
MAX_JOBS=4 bash deploy/orin_nx/setup.sh
```

This builds:

- `depthart_selective_scan_cuda` for SM87, used by PyTorch FP32/TF32 and AMP;
- `libdepthart_selective_scan_trt.so`, used by TensorRT FP32 and FP16.

For PyTorch-only inference, TensorRT plugin compilation can be skipped:

```bash
bash deploy/orin_nx/setup.sh pytorch-only
```

Verify the installed runtime:

```bash
python3 -c 'import torch, tensorrt; print(torch.__version__, torch.version.cuda, tensorrt.__version__)'
```

## 4. Run Inference

Relative S224:

```bash
python relative/infer_image.py \
  --image /path/to/image.jpg \
  --encoder S --resolution 224 \
  --checkpoint checkpoints/relative/depthart_relative_s_224.pth \
  --output outputs/orin_relative_s_224.npy
```

Metric indoor S448:

```bash
python metric/infer_image.py \
  --image /path/to/image.jpg \
  --encoder S --domain indoor \
  --checkpoint checkpoints/metric/depthart_metric_indoor_s_448.pth \
  --intrinsics 525 525 319.5 239.5 \
  --output outputs/orin_metric_s_448.npy
```

## 5. Build Engines and Benchmark

Run the complete matrix:

```bash
bash deploy/orin_nx/benchmark.sh \
  --stage all \
  --samples 1000 \
  --warmup 200 \
  --workspace-gb 4
```

The workflow reuses the static ONNX graphs in `deploy/shared/onnx`, builds 18 Orin-local TensorRT engines, runs 36 model/backend combinations, and writes:

```text
deploy/orin_nx/results/
├── engines/
├── logs/
├── results/
├── failures.json
└── summary.csv
```

Resume an interrupted run with:

```bash
bash deploy/orin_nx/benchmark.sh --stage build
bash deploy/orin_nx/benchmark.sh --stage benchmark
bash deploy/orin_nx/benchmark.sh --stage summary
```

Existing outputs are skipped unless `--force` is passed.

## 6. Monitor the Device

Run `tegrastats` on the host while benchmarking:

```bash
sudo tegrastats
```

The benchmark's TensorRT peak-memory field only covers memory visible to the PyTorch allocator. Use `tegrastats` for total device memory, power, clocks, and temperature.

Metric benchmarks assume fixed camera intrinsics and cache DenseCameraEmbedder and DAA K/V outputs. This matches deployment with a fixed physical camera.
