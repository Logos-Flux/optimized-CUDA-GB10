#!/usr/bin/env python3
"""
Build gb10_rmsnorm kernel locally for HuggingFace Kernel Hub upload.

Since kernel-builder doesn't support cu130+aarch64 yet, we build manually
and assemble the output to match the Hub format.

Usage:
    python build_local.py
"""

import hashlib
import importlib.util
import json
import os
import platform
import shutil

import torch
from torch.utils.cpp_extension import load

# Determine build variant name
torch_version = f"torch{torch.__version__.split('+')[0].rsplit('.', 1)[0].replace('.', '')}"
cuda_version = f"cu{torch.version.cuda.replace('.', '')}"

# Check CXX11 ABI
cxx11 = "cxx11-" if torch._C._GLIBCXX_USE_CXX11_ABI else ""

machine = platform.machine()
system = platform.system().lower()

variant = f"{torch_version}-{cxx11}{cuda_version}-{machine}-{system}"
print(f"Building variant: {variant}")

# Extension name: _gb10_rmsnorm_<hash>
src_hash = hashlib.md5(open("kernel_src/rmsnorm.cu", "rb").read()).hexdigest()[:7]
ext_name = f"_gb10_rmsnorm_{src_hash}"

# Build using torch cpp_extension
print(f"Compiling {ext_name}...")
ext = load(
    name=ext_name,
    sources=[
        "torch-ext/torch_binding.cpp",
        "kernel_src/rmsnorm.cu",
    ],
    extra_include_paths=["torch-ext"],
    extra_cflags=["-O3", "-DCUDA_KERNEL"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_120,code=sm_121",
        f"-DTORCH_EXTENSION_NAME={ext_name}",
        "-DCUDA_KERNEL",
    ],
    extra_ldflags=[],
    with_cuda=True,
    verbose=True,
)

# Find the compiled .so
so_path = importlib.util.find_spec(ext_name).origin
print(f"Compiled: {so_path}")

# Assemble build output
build_dir = os.path.join("build", variant)
os.makedirs(build_dir, exist_ok=True)

# 1. Copy .so
so_dest = os.path.join(build_dir, f"{ext_name}.abi3.so")
shutil.copy2(so_path, so_dest)
print(f"Copied .so -> {so_dest}")

# 2. Create metadata.json
metadata = {"version": 1, "python-depends": []}
with open(os.path.join(build_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# 3. Create _ops.py
ops_py = f'''import torch
from . import {ext_name}
ops = torch.ops.{ext_name}

def add_op_namespace_prefix(op_name: str):
    """Prefix op by namespace."""
    return f"{ext_name}::{{op_name}}"
'''
with open(os.path.join(build_dir, "_ops.py"), "w") as f:
    f.write(ops_py)

# 4. Copy __init__.py from torch-ext/gb10_rmsnorm/
# When loaded via get_kernel, the build directory IS the package root.
# __init__.py imports from ._ops which imports the .so
shutil.copy2("torch-ext/gb10_rmsnorm/__init__.py", os.path.join(build_dir, "__init__.py"))

print(f"\nBuild output: {build_dir}/")
for f in sorted(os.listdir(build_dir)):
    fpath = os.path.join(build_dir, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {f}  ({size:,} bytes)")
    else:
        print(f"  {f}/")

print("\nDone! Upload with: huggingface-cli upload logos-flux/gb10-rmsnorm ./build")
