"""
Build script for GB10-optimized CUDA kernels.

Usage:
    pip install -e .
    # or
    uv pip install -e .
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# GB10 Blackwell compute capability
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")

setup(
    name="gb10_kernels",
    version="0.1.0",
    description="Optimized CUDA kernels for NVIDIA GB10 Blackwell",
    packages=find_packages(where="torch_ext"),
    package_dir={"": "torch_ext"},
    ext_modules=[
        CUDAExtension(
            name="gb10_kernels._C",
            sources=[
                "torch_ext/torch_binding.cpp",
                "kernels/rmsnorm.cu",
                "kernels/gelu.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_120,code=sm_121",
                    "--threads=4",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch"],
)
