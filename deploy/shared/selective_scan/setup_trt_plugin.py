import os
import sys
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
SCAN_ROOT = ROOT / "vendor" / "mamba_ssm_csrc" / "selective_scan"


def first_existing(candidates, description):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate).resolve()
    checked = "\n  ".join(str(item) for item in candidates if item)
    raise RuntimeError(f"cannot find {description}; checked:\n  {checked}")


try:
    import tensorrt

    TRT_PACKAGE = Path(tensorrt.__file__).resolve().parent
except ImportError:
    TRT_PACKAGE = Path(sys.prefix) / "lib" / "python"

include_candidates = (
    os.environ.get("TENSORRT_INCLUDE_DIR"),
    "/usr/include/aarch64-linux-gnu",
    "/usr/include/x86_64-linux-gnu",
    "/usr/include",
    TRT_PACKAGE / "include",
)
TRT_INCLUDE = first_existing(
    (candidate for candidate in include_candidates if candidate and (Path(candidate) / "NvInfer.h").is_file()),
    "NvInfer.h (set TENSORRT_INCLUDE_DIR)",
)

TRT_LIBS = first_existing(
    (
        os.environ.get("TENSORRT_LIB_DIR"),
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
        TRT_PACKAGE / "../tensorrt_libs",
        TRT_PACKAGE / "lib",
    ),
    "TensorRT library directory",
)


def find_library(environment_name, stem):
    override = os.environ.get(environment_name)
    if override:
        return first_existing((override,), stem)
    return first_existing(
        tuple(TRT_LIBS.glob(f"{stem}.so*")) + tuple(Path("/usr/lib").glob(f"**/{stem}.so*")),
        stem,
    )


NVINFER_LIBRARY = find_library("TENSORRT_NVINFER_LIBRARY", "libnvinfer")
NVINFER_PLUGIN_LIBRARY = find_library("TENSORRT_PLUGIN_LIBRARY", "libnvinfer_plugin")
TORCH_LIBS = Path(torch.__file__).resolve().parent / "lib"
CXX_STANDARD = os.environ.get("DEPTHART_CXX_STANDARD", "c++17")

extension = CUDAExtension(
    name="depthart_selective_scan_trt_plugin",
    sources=[
        str(ROOT / "trt_plugin" / "selective_scan_plugin.cpp"),
        str(SCAN_ROOT / "cusoflex" / "selective_scan_core_fwd.cu"),
    ],
    include_dirs=[
        str(TRT_INCLUDE),
        str(SCAN_ROOT),
        str(SCAN_ROOT / "cusoflex"),
    ],
    extra_compile_args={
        "cxx": ["-O3", f"-std={CXX_STANDARD}", "-Wno-deprecated-declarations"],
        "nvcc": [
            "-O3",
            f"-std={CXX_STANDARD}",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
        ],
    },
    extra_link_args=[
        str(NVINFER_LIBRARY),
        str(NVINFER_PLUGIN_LIBRARY),
        f"-Wl,-rpath,{TRT_LIBS}",
        f"-Wl,-rpath,{TORCH_LIBS}",
    ],
)

setup(
    name="depthart-selective-scan-trt-plugin",
    version="0.1.0",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
