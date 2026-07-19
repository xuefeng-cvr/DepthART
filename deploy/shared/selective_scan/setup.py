import os
import subprocess
from pathlib import Path

from packaging.version import Version, parse
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "vendor" / "mamba_ssm_csrc" / "selective_scan"


def cuda_version():
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set; a CUDA toolkit is required")
    output = subprocess.check_output([str(Path(CUDA_HOME) / "bin" / "nvcc"), "-V"], text=True)
    tokens = output.split()
    return parse(tokens[tokens.index("release") + 1].rstrip(","))


def nvcc_flags(version):
    default_standard = "c++17" if version >= Version("11.0") else "c++14"
    standard = os.getenv("DEPTHART_CXX_STANDARD", default_standard)
    if standard not in ("c++14", "c++17"):
        raise RuntimeError("DEPTHART_CXX_STANDARD must be c++14 or c++17")
    flags = [
        "-O3",
        f"-std={standard}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
    ]
    if version >= Version("11.2"):
        flags.extend(("--threads", os.getenv("DEPTHART_NVCC_THREADS", "4")))
    return flags, standard


version = cuda_version()
cuda_flags, cpp_standard = nvcc_flags(version)
extension = CUDAExtension(
    name="depthart_selective_scan_cuda",
    sources=[
        str(CSRC / "cusoflex" / "selective_scan_oflex.cpp"),
        str(CSRC / "cusoflex" / "selective_scan_core_fwd.cu"),
        str(CSRC / "cusoflex" / "selective_scan_core_bwd.cu"),
    ],
    include_dirs=[str(CSRC)],
    extra_compile_args={
        "cxx": ["-O3", f"-std={cpp_standard}"],
        "nvcc": cuda_flags,
    },
)

setup(
    name="depthart-selective-scan",
    version="0.1.0",
    description="Portable mixed-precision Selective Scan operator for DepthART",
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.8",
    install_requires=["torch", "packaging"],
)
