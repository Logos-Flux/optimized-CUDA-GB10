FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create venv to avoid PEP 668 issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 13 support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Install kernel dev dependencies
RUN pip install --no-cache-dir \
    ninja \
    triton \
    pybind11 \
    numpy \
    pytest

# Set CUDA arch for GB10 (Blackwell sm_121)
ENV TORCH_CUDA_ARCH_LIST="12.1a"

WORKDIR /workspace

# Copy project files
COPY . /workspace

# Build and install kernels
RUN pip install -e .

CMD ["bash"]
