# ============================================================
# optimized-CUDA-GB10 — NGC PyTorch base for DGX Spark / GB10
# ============================================================
# Why NGC instead of nvidia/cuda + the cu130 PyTorch wheel?
# See README "Why NGC?" — short version:
#   * the official PyTorch cu130 wheel still links libcudart.so.12
#     (won't even start on a CUDA-13 system)
#   * its PTX tops out at compute capability 12.0; GB10 is 12.1, so
#     any BF16 matmul raises CUBLAS_STATUS_INVALID_VALUE
# NGC PyTorch is built by NVIDIA against CUDA 13 and includes
# CC 12.1 PTX, so it actually runs on GB10.
#
# Build:  DOCKER_BUILDKIT=1 docker build -t gb10-kernels:latest .
# Run:    docker run --rm --gpus all gb10-kernels:latest \
#             python /workspace/verify_setup.py
# ============================================================

ARG NGC_VERSION=25.12
FROM nvcr.io/nvidia/pytorch:${NGC_VERSION}-py3

# GB10 Blackwell sm_121 only — keep the kernel binary small and
# the build fast. Override at build time if you need broader compat.
ENV TORCH_CUDA_ARCH_LIST="12.1a"
ENV CUDA_HOME=/usr/local/cuda

# Performance defaults that match what our internal images run with.
# expandable_segments avoids fragmentation on the GB10's unified memory;
# the 4 GiB JIT cache prevents repeated nvcc compiles for Triton kernels.
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV CUDNN_BENCHMARK=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
ENV CUDA_CACHE_MAXSIZE=4294967296
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Build the kernels in a scratch dir, NOT in /workspace, so a runtime
# bind-mount on /workspace doesn't shadow the installed package.
WORKDIR /opt/gb10-kernels
COPY setup.py pyproject.toml ./
COPY kernels/ kernels/
COPY torch_ext/ torch_ext/

# --no-build-isolation is critical: it preserves the NGC PyTorch instead
# of letting pip pull a fresh torch wheel into an isolated build env.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --no-build-isolation .

# Sanity scripts and tests live in /workspace so they're easy to run
# (and a bind-mount can override them with newer copies).
WORKDIR /workspace
COPY verify_setup.py ./
COPY tests/ tests/
COPY benchmarks/ benchmarks/

CMD ["python", "verify_setup.py"]
