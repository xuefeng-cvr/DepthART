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
  <a href="https://arxiv.org/abs/2607.17099">
    <img src="https://img.shields.io/badge/arXiv-2607.17099-b31b1b?logo=arxiv&logoColor=white" alt="arXiv 2607.17099">
  </a>
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

## 🔎 Overview

DepthART scales foundation monocular depth estimation to compact Small, Base,
and Large models. The release supports both affine-invariant relative depth and
camera-aware metric depth, with optimized inference paths for desktop NVIDIA
GPUs and Jetson Orin NX.

| Task | Input | Output | Released variants |
| --- | --- | --- | --- |
| Relative depth | RGB image | Affine-invariant depth | TinyViM S/B/L and MobileNetV4 variants at 224 and 448 |
| Metric depth | RGB image and camera intrinsics | Depth in meters | Indoor and outdoor S/B/L at 448 |
| Deployment | Static model input | PyTorch or TensorRT inference | FP32/TF32, AMP, and FP16 |

## 📰 News

- **September 2026:** We released the MobileNetV4-S and MobileNetV4-M DepthART
  variants for deployment on legacy and resource-constrained mobile devices,
  including earlier platforms using BPU or NPU accelerators. We also provide
  MobileNetV4-M-slim-SPF, which prunes the MobileNetV4-M backbone and replaces
  the DPT decoder with our computation-efficient Single-Path Pyramid Fusion
  (SPF) decoder, offering a stronger balance between depth quality and inference
  speed.

## ✨ Highlights

- Relative depth at 224x224 and 448x448 input resolutions.
- TinyViM S/B/L, MobileNetV4-S/M, and MobileNetV4-M-slim-SPF encoders.
- Indoor and outdoor metric-depth models with camera-intrinsics conditioning.
- PyTorch FP32/TF32 and AMP inference.
- TensorRT FP32 and FP16 deployment with a custom Selective Scan plugin.
- Reproducible PC and Jetson Orin NX setup scripts.

<a id="to-do-list"></a>
## 📋 To-do List

The current public release provides inference, evaluation, and deployment code.
Training code and reproducible training configurations are not included yet.

- [ ] Release the complete training code and reproducible training configurations.
- [ ] Complete end-to-end testing on Jetson Nano and publish deployment instructions and benchmark results.
- [ ] Release the DepthART mobile application.
- [x] Add deployment-friendly MobileNetV4 DepthART variants.

## 🔗 Quick Links

- [Pretrained models](https://huggingface.co/Fengxue93/DepthART)
- [Project page](https://xuefeng-cvr.github.io/DepthART/)
- [Installation](#installation)
- [Relative and metric inference](#inference)
- [Dataset evaluation](#dataset-evaluation)
- [Accuracy and A6000 performance](#results)
- [PC and Orin NX deployment](deploy/README.md)
- [To-do list](#to-do-list)
- [Citation](#citation)

## 🗂️ Repository Layout

```text
DepthART/
├── assets/                   # README media
├── checkpoints/              # pretrained Relative and Metric models
├── relative/                 # relative-depth models, inference, and evaluation
├── metric/                   # metric-depth inference and evaluation
├── deploy/
│   ├── shared/
│   │   ├── onnx/             # complete DepthART ONNX graphs
│   │   ├── selective_scan/   # CUDA operator and TensorRT plugin
│   │   └── tests/            # full-model deployment integration tests
│   ├── pc/                   # desktop/server NVIDIA GPU workflow
│   └── orin_nx/              # Jetson Orin NX workflow
├── environment.yml           # reproducible PC Conda environment
└── CHECKSUMS.sha256          # checkpoint and ONNX checksums
```

## 🧩 Models

| Task | Domain | Scale | Input | Checkpoint |
| --- | --- | --- | --- | --- |
| Relative | general | S/B/L | 224x224 | `checkpoints/relative/depthart_relative_<s,b,l>_224.pth` |
| Relative | general | S/B/L | 448x448 | `checkpoints/relative/depthart_relative_<s,b,l>_448.pth` |
| Relative | general | MobileNetV4-S/M | 224x224 or 448x448 | `checkpoints/relative/depthart_relative_mnv4<s,m>_<224,448>.pth` |
| Relative | general | MobileNetV4-M-slim-SPF | 224x224 or 448x448 | `checkpoints/relative/depthart_relative_mnv4m_slim_<224,448>.pth` |
| Metric | indoor | S/B/L | 448 | `checkpoints/metric/depthart_metric_indoor_<s,b,l>_448.pth` |
| Metric | outdoor | S/B/L | 448 | `checkpoints/metric/depthart_metric_outdoor_<s,b,l>_448.pth` |

## 📦 Pretrained Checkpoints

Pretrained Relative and Metric checkpoints are hosted in the
[DepthART Hugging Face repository](https://huggingface.co/Fengxue93/DepthART).
Model checkpoints and ONNX graphs are not stored in the GitHub repository;
download them from Hugging Face after cloning the source code.

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
not distributed as checkpoints. `CHECKSUMS.sha256` covers all 18 checkpoints
and all 18 ONNX graphs, including the six standard-operator MobileNetV4 graphs.

Each checkpoint is an inference-only PyTorch payload with two top-level fields:
`model` contains the model state dictionary, and `validation_metrics` contains
only the validation summaries available for that model. Checkpoints do not
include optimizer state, training arguments, epochs, dataset split entries,
image paths, or internal filesystem paths.

<a id="installation"></a>
## 🛠️ Installation

### 🖥️ PC

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

### 🚀 Jetson Orin NX

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

<a id="inference"></a>
## 🔮 Inference

Both inference commands save a float32 NumPy depth map and a colorized PNG.

### 📐 Relative Depth

```bash
python relative/infer_image.py \
  --image assets/example.png \
  --encoder S \
  --resolution 224 \
  --checkpoint checkpoints/relative/depthart_relative_s_224.pth \
  --output outputs/relative_s_224.npy
```

Relative predictions are affine-invariant and should not be interpreted as metric distance.

MobileNetV4 models use the same command with one of `MNV4-S`, `MNV4-M`, or
`MNV4-M-SLIM-SPF`:

```bash
python relative/infer_image.py \
  --image assets/example.png \
  --encoder MNV4-M-SLIM-SPF \
  --resolution 448 \
  --checkpoint checkpoints/relative/depthart_relative_mnv4m_slim_448.pth \
  --output outputs/relative_mnv4m_slim_448.npy
```

The released MobileNetV4 variants support PyTorch inference and portable,
standard-operator ONNX export. The TensorRT benchmark matrix and custom
Selective Scan plugin apply to the TinyViM S/B/L models.

### 📏 Metric Depth

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

<a id="dataset-evaluation"></a>
## 🧪 Dataset Evaluation

Set `DEPTHART_DATA_ROOT` to the directory containing the supported zero-shot datasets:

```bash
export DEPTHART_DATA_ROOT=/path/to/Zero_shot_Datasets
python deploy/shared/evaluate_depth.py --output-dir outputs/evaluation
```

Relative evaluation performs per-image scale-and-shift alignment. Metric evaluation uses absolute depth without alignment. The deployment evaluator supports NYUD and KITTI for all released TinyViM S/B/L variants.

MobileNetV4 checkpoints can be evaluated directly with the relative-depth evaluator:

```bash
cd relative
python infer_dataset.py \
  --encoder MNV4-M --input_height 448 --input_width 448 \
  --pretrained_from ../checkpoints/relative/depthart_relative_mnv4m_448.pth \
  --eval_datasets NYU KITTI \
  --kitti-tiled --tile-stride 128 \
  --path-map /path/to/Zero_shot_Datasets="$DEPTHART_DATA_ROOT" \
  --output-json ../outputs/mnv4m_448.json
```

KITTI is resized with aspect ratio preserved and evaluated with square
horizontal windows. The windows are jointly scale-and-shift aligned from their
overlaps only, blended with a Hann window, and then evaluated with one
per-image disparity affine alignment inside the Eigen crop. Use window/stride
`448/128` for 448 checkpoints and `224/64` for 224 checkpoints. Ground truth is
never used to stitch the windows.

<a id="results"></a>
## 📊 Results

### 🎯 Depth Accuracy

Relative-depth results use per-image affine scale-and-shift alignment on the valid ground-truth mask. Metric-depth results use absolute depth without alignment; NYUD uses the indoor checkpoint and KITTI uses the outdoor checkpoint. `delta1` is higher-is-better, while AbsRel and RMSE are lower-is-better.

#### 📐 Relative Depth

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

The released MobileNetV4 checkpoints use affine-consistent tiled KITTI
inference. NYUD uses the native full-image path. All numbers below were
reproduced on the complete 654-image NYUD and 652-image KITTI splits:

| Model | Input | NYUD delta1 | NYUD AbsRel | KITTI delta1 | KITTI AbsRel |
| --- | --- | ---: | ---: | ---: | ---: |
| MobileNetV4-S | 224 | 0.922 | 0.088 | 0.890 | 0.101 |
| MobileNetV4-S | 448 | 0.934 | 0.082 | 0.910 | 0.091 |
| MobileNetV4-M | 224 | 0.944 | 0.073 | 0.922 | 0.085 |
| MobileNetV4-M | 448 | 0.953 | 0.067 | 0.931 | 0.080 |
| MobileNetV4-M-slim-SPF | 224 | 0.942 | 0.073 | 0.920 | 0.085 |
| MobileNetV4-M-slim-SPF | 448 | 0.952 | 0.068 | 0.928 | 0.082 |

The complete records are in [`mobilenetv4_relative_accuracy.csv`](deploy/pc/results/a6000/mobilenetv4_relative_accuracy.csv).

##### 🆕 Latest Tiny Depth Model Comparison (September 2026)

All entries use per-image affine alignment in disparity space. Each cell is
`delta1 / AbsRel` (higher/lower is better). TinyViM rows are the reported
DepthART results, while the other rows were evaluated over every image in each
listed split.

| Model | NYUD | KITTI | ETH3D | DDAD | DIODE |
| --- | ---: | ---: | ---: | ---: | ---: |
| DepthAnything V2-S [[1]](#tiny-model-ref-1) | **0.974 / 0.051** | **0.944 / 0.077** | **0.964 / 0.065** | **0.921 / 0.085** | 0.756 / 0.207 |
| YOLO26-N-Depth [[2]](#tiny-model-ref-2) | 0.875 / 0.112 | 0.767 / 0.158 | 0.876 / 0.121 | 0.793 / 0.168 | 0.718 / 0.221 |
| YOLO26-S-Depth [[2]](#tiny-model-ref-2) | 0.887 / 0.108 | 0.789 / 0.151 | 0.878 / 0.118 | 0.777 / 0.174 | 0.718 / 0.218 |
| YOLO26-M-Depth [[2]](#tiny-model-ref-2) | 0.931 / 0.087 | 0.838 / 0.135 | 0.910 / 0.102 | 0.829 / 0.150 | 0.756 / 0.201 |
| YOLO26-L-Depth [[2]](#tiny-model-ref-2) | 0.939 / 0.082 | 0.863 / 0.118 | 0.915 / 0.099 | 0.833 / 0.150 | **0.768 / 0.192** |
| YOLO26-X-Depth [[2]](#tiny-model-ref-2) | 0.943 / 0.079 | 0.847 / 0.124 | 0.932 / 0.089 | 0.840 / 0.143 | 0.766 / 0.193 |
| ZipDepth Base [[3]](#tiny-model-ref-3) | 0.935 / 0.082 | 0.881 / 0.114 | 0.927 / 0.101 | 0.892 / 0.102 | 0.731 / 0.224 |
| DepthART MobileNetV4-S | 0.934 / 0.082 | 0.910 / 0.091<sup>SW</sup> | 0.906 / 0.112 | 0.860 / 0.123 | 0.714 / 0.226 |
| DepthART MNv4-M | 0.953 / 0.067 | 0.931 / 0.080<sup>SW</sup> | 0.930 / 0.095 | 0.887 / 0.106 | 0.732 / 0.217 |
| **DepthART MNv4-M-slim-SPF**⭐ | 0.952 / 0.068 | 0.928 / 0.082<sup>SW</sup> | 0.928 / 0.098 | 0.879 / 0.115 | 0.732 / 0.216 |
| DepthART TinyViM-S | 0.964 / 0.059 | 0.930 / 0.082 | 0.950 / 0.084 | 0.900 / 0.095 | 0.745 / 0.214 |
| DepthART TinyViM-B | 0.969 / 0.057 | 0.929 / 0.088 | 0.954 / 0.092 | 0.906 / 0.094 | 0.746 / 0.214 |
| DepthART TinyViM-L | 0.971 / 0.053 | 0.933 / 0.079 | 0.958 / 0.091 | 0.910 / 0.095 | 0.754 / 0.208 |

<sup>SW</sup> MobileNetV4 KITTI results use sliding-window inference. These
backbones are sensitive to test-image aspect ratios that differ substantially
from the training crops, so direct full-width inference can reduce accuracy.

<a id="tiny-model-ref-1"></a>[1] Yang et al., [*Depth Anything V2*](https://arxiv.org/abs/2406.09414), 2024.<br>
<a id="tiny-model-ref-2"></a>[2] Jocher et al., [*Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*](https://arxiv.org/abs/2606.03748), 2026.<br>
<a id="tiny-model-ref-3"></a>[3] Tosi et al., [*ZipDepth: Bringing Lightweight Zero-Shot Monocular Depth Anywhere, on Any Device*](https://arxiv.org/abs/2607.08771), ECCV 2026.

The unrounded D1, D2, D3, AbsRel, SqRel, RMSE, RMSE-log, log10, and SILog
records are available in [`complete_depth_accuracy.csv`](deploy/pc/results/a6000/complete_depth_accuracy.csv).

##### 🖼️ Qualitative Comparison

Depth maps use the Spectral colormap (near is red, far is blue). Benchmark
visualizations are affine-aligned to ground truth; in-the-wild outputs use the
same per-image percentile normalization for every model. The displayed maps
preserve the source aspect ratio; they are not forced to square inputs. The
GMAC values use the conventional square reference tensor shown after `@`.
ZipDepth uses `short=448` in this matched comparison, while its official
default is `short=384` (approximately 3.31 GMAC at 384x384).

| DepthAnything V2-S<br><sub>short=518, aspect ratio preserved · 24.79M params · 41.30 GMAC @ 518x518</sub> | YOLO26-L-Depth<br><sub>imgsz=768, letterboxed · 27.68M params · 78.50 GMAC @ 768x768</sub> | ZipDepth<br><sub>short=448, aspect ratio preserved · 6.14M params · 4.50 GMAC @ 448x448</sub> |
| --- | --- | --- |
| <img src="assets/comparison/kitti/depthanything-v2-s.jpg" width="100%"> | <img src="assets/comparison/kitti/yolo26-l-depth.jpg" width="100%"> | <img src="assets/comparison/kitti/zipdepth.jpg" width="100%"> |
| DepthART TinyViM-S<br><sub>short=448, aspect ratio preserved · 6.03M params · 7.00 GMAC @ 448x448</sub> | DepthART MNv4-M<br><sub>448x448 sliding windows · 8.55M params · 8.63 GMAC/window</sub> | ⭐ **DepthART MNv4-M-slim-SPF**<br><sub>448x448 sliding windows · 6.05M params · 3.78 GMAC/window</sub> |
| <img src="assets/comparison/kitti/depthart-tinyvim-s.jpg" width="100%"> | <img src="assets/comparison/kitti/depthart-mnv4-m.jpg" width="100%"> | <img src="assets/comparison/kitti/depthart-mnv4-m-slim-spf.jpg" width="100%"> |

<p align="center"><sub>KITTI validation sample 000000048</sub></p>

| DepthAnything V2-S<br><sub>short=518, aspect ratio preserved · 24.79M params · 41.30 GMAC @ 518x518</sub> | YOLO26-L-Depth<br><sub>imgsz=768, letterboxed · 27.68M params · 78.50 GMAC @ 768x768</sub> | ZipDepth<br><sub>short=448, aspect ratio preserved · 6.14M params · 4.50 GMAC @ 448x448</sub> |
| --- | --- | --- |
| <img src="assets/comparison/self-collection-001/depthanything-v2-s.jpg" width="100%"> | <img src="assets/comparison/self-collection-001/yolo26-l-depth.jpg" width="100%"> | <img src="assets/comparison/self-collection-001/zipdepth.jpg" width="100%"> |
| DepthART TinyViM-S<br><sub>short=448, aspect ratio preserved · 6.03M params · 7.00 GMAC @ 448x448</sub> | DepthART MNv4-M<br><sub>short=448, aspect ratio preserved · 8.55M params · 8.63 GMAC @ 448x448</sub> | **DepthART MNv4-M-slim-SPF**<br><sub>short=448, aspect ratio preserved · 6.05M params · 3.78 GMAC @ 448x448</sub> |
| <img src="assets/comparison/self-collection-001/depthart-tinyvim-s.jpg" width="100%"> | <img src="assets/comparison/self-collection-001/depthart-mnv4-m.jpg" width="100%"> | <img src="assets/comparison/self-collection-001/depthart-mnv4-m-slim-spf.jpg" width="100%"> |

<p align="center"><sub>Self-collected sample 001</sub></p>

| DepthAnything V2-S<br><sub>short=518 · 24.79M · 41.30 GMAC @ 518x518</sub> | YOLO26-L-Depth<br><sub>imgsz=768 · 27.68M · 78.50 GMAC @ 768x768</sub> | ZipDepth<br><sub>short=448 · 6.14M · 4.50 GMAC @ 448x448</sub> | DepthART TinyViM-S<br><sub>short=448 · 6.03M · 7.00 GMAC @ 448x448</sub> | DepthART MNv4-M<br><sub>short=448 · 8.55M · 8.63 GMAC @ 448x448</sub> | **DepthART MNv4-M-slim-SPF**<br><sub>short=448 · 6.05M · 3.78 GMAC @ 448x448</sub> |
| --- | --- | --- | --- | --- | --- |
| <img src="assets/comparison/self-collection-009/depthanything-v2-s.jpg" width="100%"> | <img src="assets/comparison/self-collection-009/yolo26-l-depth.jpg" width="100%"> | <img src="assets/comparison/self-collection-009/zipdepth.jpg" width="100%"> | <img src="assets/comparison/self-collection-009/depthart-tinyvim-s.jpg" width="100%"> | <img src="assets/comparison/self-collection-009/depthart-mnv4-m.jpg" width="100%"> | <img src="assets/comparison/self-collection-009/depthart-mnv4-m-slim-spf.jpg" width="100%"> |

<p align="center"><sub>Self-collected sample 009</sub></p>

| DepthAnything V2-S<br><sub>short=518, aspect ratio preserved · 24.79M params · 41.30 GMAC @ 518x518</sub> | YOLO26-L-Depth<br><sub>imgsz=768, letterboxed · 27.68M params · 78.50 GMAC @ 768x768</sub> | ZipDepth<br><sub>short=448, aspect ratio preserved · 6.14M params · 4.50 GMAC @ 448x448</sub> |
| --- | --- | --- |
| <img src="assets/comparison/self-collection-019/depthanything-v2-s.jpg" width="100%"> | <img src="assets/comparison/self-collection-019/yolo26-l-depth.jpg" width="100%"> | <img src="assets/comparison/self-collection-019/zipdepth.jpg" width="100%"> |
| DepthART TinyViM-S<br><sub>short=448, aspect ratio preserved · 6.03M params · 7.00 GMAC @ 448x448</sub> | DepthART MNv4-M<br><sub>short=448, aspect ratio preserved · 8.55M params · 8.63 GMAC @ 448x448</sub> | **DepthART MNv4-M-slim-SPF**<br><sub>short=448, aspect ratio preserved · 6.05M params · 3.78 GMAC @ 448x448</sub> |
| <img src="assets/comparison/self-collection-019/depthart-tinyvim-s.jpg" width="100%"> | <img src="assets/comparison/self-collection-019/depthart-mnv4-m.jpg" width="100%"> | <img src="assets/comparison/self-collection-019/depthart-mnv4-m-slim-spf.jpg" width="100%"> |

<p align="center"><sub>Self-collected sample 019</sub></p>

| DepthAnything V2-S<br><sub>short=518 · 24.79M · 41.30 GMAC @ 518x518</sub> | YOLO26-L-Depth<br><sub>imgsz=768 · 27.68M · 78.50 GMAC @ 768x768</sub> | ZipDepth<br><sub>short=448 · 6.14M · 4.50 GMAC @ 448x448</sub> | DepthART TinyViM-S<br><sub>short=448 · 6.03M · 7.00 GMAC @ 448x448</sub> | DepthART MNv4-M<br><sub>short=448 · 8.55M · 8.63 GMAC @ 448x448</sub> | **DepthART MNv4-M-slim-SPF**<br><sub>short=448 · 6.05M · 3.78 GMAC @ 448x448</sub> |
| --- | --- | --- | --- | --- | --- |
| <img src="assets/comparison/self-collection-029/depthanything-v2-s.jpg" width="100%"> | <img src="assets/comparison/self-collection-029/yolo26-l-depth.jpg" width="100%"> | <img src="assets/comparison/self-collection-029/zipdepth.jpg" width="100%"> | <img src="assets/comparison/self-collection-029/depthart-tinyvim-s.jpg" width="100%"> | <img src="assets/comparison/self-collection-029/depthart-mnv4-m.jpg" width="100%"> | <img src="assets/comparison/self-collection-029/depthart-mnv4-m-slim-spf.jpg" width="100%"> |

<p align="center"><sub>Self-collected sample 029</sub></p>

#### 📏 Metric Depth

| Model | Dataset | Actual input | TF32 delta1 | TF32 RMSE (m) | TRT FP32 delta1 | TRT FP32 RMSE (m) | TRT FP16 delta1 | TRT FP16 RMSE (m) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S448 | NYUD | 480x640 | 0.9234 | 0.3353 | 0.9234 | 0.3355 | 0.9232 | 0.3364 |
| S448 | KITTI | 448x1472 | 0.9489 | 3.0189 | 0.9490 | 3.0165 | 0.9483 | 3.0270 |
| B448 | NYUD | 480x640 | 0.9420 | 0.3075 | 0.9421 | 0.3074 | 0.9420 | 0.3071 |
| B448 | KITTI | 448x1472 | 0.9499 | 2.9462 | 0.9499 | 2.9455 | 0.9502 | 2.9415 |
| L448 | NYUD | 480x640 | 0.9459 | 0.2953 | 0.9459 | 0.2952 | 0.9439 | 0.3006 |
| L448 | KITTI | 448x1472 | 0.9575 | 2.7652 | 0.9576 | 2.7640 | 0.9583 | 2.7663 |

The complete per-backend accuracy records, checkpoint paths, engine paths, sample counts, and evaluation protocols are available in [`backend_depth_metrics.csv`](deploy/pc/results/a6000/depth_eval/backend_depth_metrics.csv). Rounded PyTorch checkpoint results are retained in [`depth_metrics.csv`](deploy/pc/results/a6000/depth_eval/depth_metrics.csv).

### ⚡ A6000 Inference Performance

The following results were measured on an NVIDIA RTX A6000 with batch size 1, 200 warm-up iterations, and 1000 timed samples. Model-only latency uses CUDA events and excludes image decoding, CPU preprocessing, host-to-device transfer, output transfer, and output resize. FPS is `1000 / model-only latency (ms)`. End-to-end latency is wall-clock time from a deterministic 640x480 RGB sample through CPU resize and normalization, host-to-device transfer, inference, device-to-host transfer, and depth resize back to 640x480.

#### 📐 Relative Depth

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
| MobileNetV4-S-448 | PyTorch FP32 (TF32 off) | 2.306 | 433.7 | 5.135 | 21.8 |
| MobileNetV4-M-448 | PyTorch FP32 (TF32 off) | 3.942 | 253.7 | 6.793 | 46.3 |
| MobileNetV4-M-slim-SPF-448 | PyTorch FP32 (TF32 off) | 2.709 | 369.1 | 5.638 | 38.3 |

#### 📏 Metric Depth

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
The MobileNetV4 FP32 records are available in [`mobilenetv4_relative_speed.csv`](deploy/pc/results/a6000/mobilenetv4_relative_speed.csv).

## ⏱️ Performance Benchmark

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

## ✅ Verify the Repository

```bash
python verify_release.py --allow-build-artifacts
```

This checks both SHA256 manifests, the checkpoint inventory, result tables,
ONNX graphs, and custom Selective Scan nodes while allowing libraries produced
locally by the setup scripts. Release maintainers should run
`python verify_release.py` without the flag on a clean source tree before
publishing.

<a id="citation"></a>
## 📝 Citation

If you find DepthART useful in your research, please consider citing:

```bibtex
@inproceedings{depthart2026,
  title     = {DepthART: Scaling Foundation Monocular Depth to Tiny Models},
  author    = {Feng Xue and Wu Chen and Mingshuai Zhao and Guofeng Zhong and Anlong Ming and Haozhe Wang and Dianqiao Lei and Zhaowen Lin and Haiyang Zhang and Nicu Sebe},
  booktitle = {ACM Multimedia},
  year      = {2026}
}
```

## ⚖️ License and Third-Party Notices

Original DepthART materials authored by this project are released under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0). This license applies only to material owned by the DepthART authors
and does not relicense third-party code, models, weights, datasets, or assets.

Some included or referenced materials use different terms. In particular,
`metric/network/attention.py` is marked CC BY-NC 4.0; the vendored Selective
Scan implementation retains its upstream copyright and license terms; and the
DepthART checkpoints were trained using supervision generated with
[Depth Anything V2-L](https://github.com/DepthAnything/Depth-Anything-V2),
whose model weights are distributed under CC BY-NC 4.0. These
materials are not covered by the DepthART CC BY 4.0 grant. Users must review
and comply with all applicable third-party licenses, including restrictions on
commercial use, before using or redistributing the corresponding code or
weights. See [`LICENSE`](LICENSE) for the project license notice.
