<p align="right">
  <a href="README.md"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="README_zh-CN.md"><img src="https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-lightgrey" alt="中文"></a>
</p>

<div align="center">

<h1>DepthART: Scaling Foundation Monocular Depth to Tiny Models</h1>

<p><strong>ACM Multimedia 2026</strong></p>

<p>
  <a href="https://xuefeng-cvr.github.io/DepthART/">
    <img src="https://img.shields.io/badge/Project-Page-red?logo=googlechrome&logoColor=white" alt="Project page">
  </a>
  <a href="https://huggingface.co/Fengxue93/DepthART">
    <img src="https://img.shields.io/badge/Hugging_Face-Models-yellow?logo=huggingface&logoColor=black" alt="Hugging Face models">
  </a>
  <img src="https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b?logo=arxiv&logoColor=white" alt="arXiv coming soon">
</p>

<p>
  <em>Feng Xue</em> &bull; Wu Chen &bull; Mingshuai Zhao &bull; Guofeng Zhong &bull; Anlong Ming<br>
  Haozhe Wang &bull; Dianqiao Lei &bull; Zhaowen Lin &bull; Haiyang Zhang &bull; Nicu Sebe
</p>

<a href="depthart-intro.mp4">
  <img src="assets/depthart-intro.gif" alt="DepthART introduction and qualitative results" width="95%">
</a>

<sub>Click the preview to open the full-resolution MP4.</sub>

</div>

## Overview

DepthART scales foundation monocular depth estimation to compact Small, Base,
and Large models. The release supports both affine-invariant relative depth and
camera-aware metric depth, with optimized inference paths for desktop NVIDIA
GPUs and Jetson Orin NX.

| Task | Input | Output | Released variants |
| --- | --- | --- | --- |
| Relative depth | RGB image | Affine-invariant depth | S/B/L at 224 and 448 |
| Metric depth | RGB image and camera intrinsics | Depth in meters | Indoor and outdoor S/B/L at 448 |
| Deployment | Static model input | PyTorch or TensorRT inference | FP32/TF32, AMP, and FP16 |

## Highlights

- Relative depth at 224x224 and 448x448 input resolutions.
- Indoor and outdoor metric-depth models with camera-intrinsics conditioning.
- PyTorch FP32/TF32 and AMP inference.
- TensorRT FP32 and FP16 deployment with a custom Selective Scan plugin.
- Reproducible PC and Jetson Orin NX setup scripts.

## To-do List

The current public release provides inference, evaluation, and deployment code.
Training code and reproducible training configurations are not included yet.

- [ ] Release the complete training code and reproducible training configurations.
- [ ] Complete end-to-end testing on Jetson Nano and publish deployment instructions and benchmark results.
- [ ] Release the DepthART mobile application.

## Quick Links

- [Pretrained models](https://huggingface.co/Fengxue93/DepthART)
- [Project page](https://xuefeng-cvr.github.io/DepthART/)
- [Installation](#installation)
- [Relative and metric inference](#inference)
- [Dataset evaluation](#dataset-evaluation)
- [Accuracy and A6000 performance](#results)
- [PC and Orin NX deployment](deploy/README.md)
- [To-do list](#to-do-list)
- [Citation](#citation)

## Repository Layout

```text
DepthART/
├── assets/                   # README media
├── checkpoints/              # pretrained Relative and Metric models
├── relative/                 # relative-depth inference and evaluation
├── metric/                   # metric-depth inference and evaluation
├── deploy/
│   ├── shared/
│   │   ├── onnx/             # complete DepthART ONNX graphs
│   │   ├── selective_scan/   # CUDA operator and TensorRT plugin
│   │   └── tests/            # full-model deployment integration tests
│   ├── pc/                   # desktop/server NVIDIA GPU workflow
│   └── orin_nx/              # Jetson Orin NX workflow
├── environment.yml           # reproducible PC Conda environment
└── CHECKSUMS.sha256          # checkpoint checksums
```

## Models

| Task | Domain | Scale | Input | Checkpoint |
| --- | --- | --- | --- | --- |
| Relative | general | S/B/L | 224x224 | `checkpoints/relative/depthart_relative_<s,b,l>_224.pth` |
| Relative | general | S/B/L | 448x448 | `checkpoints/relative/depthart_relative_<s,b,l>_448.pth` |
| Metric | indoor | S/B/L | 448 | `checkpoints/metric/depthart_metric_indoor_<s,b,l>_448.pth` |
| Metric | outdoor | S/B/L | 448 | `checkpoints/metric/depthart_metric_outdoor_<s,b,l>_448.pth` |

## Pretrained Checkpoints

Pretrained Relative and Metric checkpoints are hosted in the
[DepthART Hugging Face repository](https://huggingface.co/Fengxue93/DepthART).

Download the complete checkpoint directory into an existing source checkout:

```bash
python -m pip install -U huggingface_hub
hf download Fengxue93/DepthART \
  --include "relative/**" \
  --include "metric/**" \
  --local-dir checkpoints
hf download Fengxue93/DepthART \
  --include "onnx/**" \
  --local-dir deploy/shared
sha256sum -c CHECKSUMS.sha256
```

The Hugging Face repository has three top-level model folders: `relative/`,
`metric/`, and `onnx/`. The commands above place them directly at the runtime
paths shown in the model table and under `deploy/shared/onnx/`, so no manual file
moves are required. TensorRT engines are device-specific build artifacts and are
not distributed as checkpoints.

Each checkpoint is an inference-only PyTorch payload with two top-level fields:
`model` contains the model state dictionary, and `validation_metrics` contains
only the validation summaries available for that model. Checkpoints do not
include optimizer state, training arguments, epochs, dataset split entries,
image paths, or internal filesystem paths.

## Installation

### PC

The tested PC configuration is defined in [`environment.yml`](environment.yml): Python 3.9, PyTorch 2.1, and CUDA 11.8.

```bash
conda env create -f environment.yml
conda activate depthart
```

Compile the PyTorch Selective Scan extension for the current GPU:

```bash
MAX_JOBS=4 bash deploy/pc/setup.sh pytorch-only
```

For TensorRT deployment, install a TensorRT development distribution that includes the Python bindings, `NvInfer.h`, `libnvinfer`, and `libnvinfer_plugin`, then run:

```bash
MAX_JOBS=4 bash deploy/pc/setup.sh
```

The pip-only TensorRT package does not provide C++ headers, so it is insufficient for compiling the custom plugin. See [PC deployment](deploy/pc/README.md) for complete requirements and troubleshooting.

### Jetson Orin NX

The recommended Orin NX environment uses JetPack 6.2 and NVIDIA's PyTorch 25.06 container. The exact container configuration is defined by [`deploy/orin_nx/Dockerfile`](deploy/orin_nx/Dockerfile).

On the Orin NX host:

```bash
cd DepthART
docker build -f deploy/orin_nx/Dockerfile -t depthart:orin-nx .

sudo nvpmodel -m 0
sudo jetson_clocks

docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v "$PWD":/workspace/DepthART \
  -w /workspace/DepthART \
  depthart:orin-nx
```

Inside the container, compile the SM87 PyTorch extension and TensorRT plugin:

```bash
MAX_JOBS=4 bash deploy/orin_nx/setup.sh
```

Do not copy CUDA extensions or TensorRT engines from a PC. Both must be built on the Orin NX against its local CUDA, PyTorch, and TensorRT versions. Continue with [Orin NX deployment](deploy/orin_nx/README.md).

## Inference

Both inference commands save a float32 NumPy depth map and a colorized PNG.

### Relative Depth

```bash
python relative/infer_image.py \
  --image assets/example.png \
  --encoder S \
  --resolution 224 \
  --checkpoint checkpoints/relative/depthart_relative_s_224.pth \
  --output outputs/relative_s_224.npy
```

Relative predictions are affine-invariant and should not be interpreted as metric distance.

### Metric Depth

Metric inference requires camera intrinsics in the order `fx fy cx cy`:

```bash
python metric/infer_image.py \
  --image assets/example.png \
  --encoder S \
  --domain indoor \
  --checkpoint checkpoints/metric/depthart_metric_indoor_s_448.pth \
  --intrinsics 525 525 319.5 239.5 \
  --output outputs/metric_indoor_s_448.npy
```

Use an indoor checkpoint for indoor cameras and an outdoor checkpoint for outdoor scenes. Intrinsics must correspond to the original input image; preprocessing scales them together with the image.

## Dataset Evaluation

Set `DEPTHART_DATA_ROOT` to the directory containing the supported zero-shot datasets:

```bash
export DEPTHART_DATA_ROOT=/path/to/Zero_shot_Datasets
python deploy/shared/evaluate_depth.py --output-dir outputs/evaluation
```

Relative evaluation performs per-image scale-and-shift alignment. Metric evaluation uses absolute depth without alignment. The evaluation script supports NYUD and KITTI for all released S/B/L variants.

## Results

### Depth Accuracy

Relative-depth results use per-image affine scale-and-shift alignment on the valid ground-truth mask. Metric-depth results use absolute depth without alignment; NYUD uses the indoor checkpoint and KITTI uses the outdoor checkpoint. `delta1` is higher-is-better, while AbsRel and RMSE are lower-is-better.

#### Relative Depth

| Model | Dataset | Actual input | TF32 delta1 | TF32 AbsRel | TRT FP32 delta1 | TRT FP32 AbsRel | TRT FP16 delta1 | TRT FP16 AbsRel |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S224 | NYUD | 224x288 | 0.9531 | 0.0664 | 0.9531 | 0.0664 | 0.9529 | 0.0666 |
| S224 | KITTI | 224x736 | 0.9218 | 0.0867 | 0.9218 | 0.0867 | 0.9217 | 0.0868 |
| S448 | NYUD | 448x608 | 0.9642 | 0.0588 | 0.9642 | 0.0588 | 0.9642 | 0.0589 |
| S448 | KITTI | 448x1472 | 0.9298 | 0.0817 | 0.9297 | 0.0818 | 0.9291 | 0.0818 |
| B224 | NYUD | 224x288 | 0.9590 | 0.0620 | 0.9590 | 0.0620 | 0.9589 | 0.0619 |
| B224 | KITTI | 224x736 | 0.9216 | 0.0923 | 0.9215 | 0.0924 | 0.9215 | 0.0922 |
| B448 | NYUD | 448x608 | 0.9691 | 0.0555 | 0.9691 | 0.0555 | 0.9690 | 0.0556 |
| B448 | KITTI | 448x1472 | 0.9297 | 0.0876 | 0.9297 | 0.0876 | 0.9292 | 0.0879 |
| L224 | NYUD | 224x288 | 0.9666 | 0.0561 | 0.9666 | 0.0561 | 0.9662 | 0.0564 |
| L224 | KITTI | 224x736 | 0.9276 | 0.0882 | 0.9277 | 0.0882 | 0.9279 | 0.0881 |
| L448 | NYUD | 448x608 | 0.9707 | 0.0539 | 0.9707 | 0.0538 | 0.9706 | 0.0540 |
| L448 | KITTI | 448x1472 | 0.9315 | 0.0842 | 0.9315 | 0.0842 | 0.9316 | 0.0836 |

Relative-L-224 TensorRT FP16 keeps the final depth prediction head in FP32. This avoids input-dependent FP16 overflow in the released checkpoint while retaining FP16 execution for the backbone and decoder.

#### Metric Depth

| Model | Dataset | Actual input | TF32 delta1 | TF32 RMSE (m) | TRT FP32 delta1 | TRT FP32 RMSE (m) | TRT FP16 delta1 | TRT FP16 RMSE (m) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S448 | NYUD | 480x640 | 0.9234 | 0.3353 | 0.9234 | 0.3355 | 0.9232 | 0.3364 |
| S448 | KITTI | 448x1472 | 0.9489 | 3.0189 | 0.9490 | 3.0165 | 0.9483 | 3.0270 |
| B448 | NYUD | 480x640 | 0.9420 | 0.3075 | 0.9421 | 0.3074 | 0.9420 | 0.3071 |
| B448 | KITTI | 448x1472 | 0.9499 | 2.9462 | 0.9499 | 2.9455 | 0.9502 | 2.9415 |
| L448 | NYUD | 480x640 | 0.9459 | 0.2953 | 0.9459 | 0.2952 | 0.9439 | 0.3006 |
| L448 | KITTI | 448x1472 | 0.9575 | 2.7652 | 0.9576 | 2.7640 | 0.9583 | 2.7663 |

The complete per-backend accuracy records, checkpoint paths, engine paths, sample counts, and evaluation protocols are available in [`backend_depth_metrics.csv`](deploy/pc/results/a6000/depth_eval/backend_depth_metrics.csv). Rounded PyTorch checkpoint results are retained in [`depth_metrics.csv`](deploy/pc/results/a6000/depth_eval/depth_metrics.csv).

### A6000 Inference Performance

The following results were measured on an NVIDIA RTX A6000 with batch size 1, 200 warm-up iterations, and 1000 timed samples. Model-only latency uses CUDA events and excludes image decoding, CPU preprocessing, host-to-device transfer, output transfer, and output resize. FPS is `1000 / model-only latency (ms)`. End-to-end latency is wall-clock time from a deterministic 640x480 RGB sample through CPU resize and normalization, host-to-device transfer, inference, device-to-host transfer, and depth resize back to 640x480.

#### Relative Depth

| Model | Backend | Model-only (ms) | FPS | E2E (ms) | Peak (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| S224 | PyTorch FP32/TF32 | 1.690 | 591.9 | 2.586 | 50.7 |
| S224 | PyTorch AMP | 2.286 | 437.4 | 3.169 | 50.6 |
| S224 | TensorRT FP32 | 1.295 | 772.5 | 2.133 | 11.0 |
| S224 | TensorRT FP16 | 0.918 | 1088.9 | 1.775 | 11.0 |
| S448 | PyTorch FP32/TF32 | 2.937 | 340.5 | 5.860 | 59.3 |
| S448 | PyTorch AMP | 3.024 | 330.7 | 5.886 | 58.9 |
| S448 | TensorRT FP32 | 2.118 | 472.2 | 5.118 | 19.6 |
| S448 | TensorRT FP16 | 1.355 | 738.0 | 4.235 | 19.6 |
| B224 | PyTorch FP32/TF32 | 1.817 | 550.3 | 2.685 | 71.2 |
| B224 | PyTorch AMP | 2.496 | 400.6 | 3.377 | 71.1 |
| B224 | TensorRT FP32 | 1.432 | 698.2 | 2.280 | 11.0 |
| B224 | TensorRT FP16 | 1.002 | 997.6 | 1.853 | 11.0 |
| B448 | PyTorch FP32/TF32 | 3.356 | 297.9 | 6.310 | 79.8 |
| B448 | PyTorch AMP | 3.406 | 293.6 | 6.308 | 79.5 |
| B448 | TensorRT FP32 | 2.429 | 411.8 | 5.559 | 19.6 |
| B448 | TensorRT FP16 | 1.485 | 673.2 | 4.342 | 19.6 |
| L224 | PyTorch FP32/TF32 | 2.448 | 408.5 | 3.341 | 152.7 |
| L224 | PyTorch AMP | 3.212 | 311.4 | 4.147 | 152.6 |
| L224 | TensorRT FP32 | 2.014 | 496.5 | 2.868 | 11.0 |
| L224 | TensorRT FP16 | 1.309 | 764.2 | 2.130 | 11.0 |
| L448 | PyTorch FP32/TF32 | 5.015 | 199.4 | 7.876 | 162.7 |
| L448 | PyTorch AMP | 4.641 | 215.5 | 7.571 | 162.4 |
| L448 | TensorRT FP32 | 3.798 | 263.3 | 6.654 | 19.6 |
| L448 | TensorRT FP16 | 2.133 | 468.8 | 4.980 | 19.6 |

#### Metric Depth

| Model | Backend | Model-only (ms) | FPS | E2E (ms) | Peak (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| S448 | PyTorch FP32/TF32 | 9.065 | 110.3 | 12.104 | 91.7 |
| S448 | PyTorch AMP | 4.750 | 210.5 | 7.620 | 91.3 |
| S448 | TensorRT FP32 | 6.364 | 157.1 | 9.301 | 19.6 |
| S448 | TensorRT FP16 | 2.558 | 391.0 | 5.425 | 19.6 |
| B448 | PyTorch FP32/TF32 | 9.858 | 101.4 | 12.829 | 121.1 |
| B448 | PyTorch AMP | 5.197 | 192.4 | 8.084 | 120.7 |
| B448 | TensorRT FP32 | 6.743 | 148.3 | 9.697 | 19.6 |
| B448 | TensorRT FP16 | 2.701 | 370.3 | 5.581 | 19.6 |
| L448 | PyTorch FP32/TF32 | 12.155 | 82.3 | 15.264 | 220.2 |
| L448 | PyTorch AMP | 6.552 | 152.6 | 9.436 | 219.9 |
| L448 | TensorRT FP32 | 8.227 | 121.6 | 11.229 | 19.6 |
| L448 | TensorRT FP16 | 3.369 | 296.8 | 6.339 | 19.6 |

Peak memory is `torch.cuda.max_memory_allocated()`. For TensorRT rows this only covers allocations visible to PyTorch and does not represent total engine or CUDA-context memory. The complete unrounded results, percentiles, software versions, validation fields, and artifact paths are available in [`summary.csv`](deploy/pc/results/a6000/summary.csv).

## Performance Benchmark

The benchmark covers PyTorch FP32/TF32, PyTorch AMP, TensorRT FP32, and TensorRT FP16. The formal protocol uses batch size 1, 200 warm-up iterations, and 1000 timed samples.

PC:

```bash
bash deploy/pc/benchmark.sh \
  --stage all --samples 1000 --warmup 200 --workspace-gb 8
```

Orin NX:

```bash
bash deploy/orin_nx/benchmark.sh \
  --stage all --samples 1000 --warmup 200 --workspace-gb 4
```

Benchmark stages can be resumed with `--stage build`, `--stage benchmark`, and `--stage summary`. See [`deploy/README.md`](deploy/README.md) for output layout and timing definitions.

## Verify the Repository

```bash
python verify_release.py --allow-build-artifacts
```

This checks the checkpoint inventory, result tables, ONNX graphs, and custom
Selective Scan nodes while allowing libraries produced locally by the setup
scripts. Release maintainers should run `python verify_release.py` without the
flag on a clean source tree before publishing.

## Citation

If you find DepthART useful in your research, please consider citing:

```bibtex
@inproceedings{depthart2026,
  title     = {DepthART: Scaling Foundation Monocular Depth to Tiny Models},
  author    = {Feng Xue and Wu Chen and Mingshuai Zhao and Guofeng Zhong and Anlong Ming and Haozhe Wang and Dianqiao Lei and Zhaowen Lin and Haiyang Zhang and Nicu Sebe},
  booktitle = {ACM Multimedia},
  year      = {2026}
}
```
