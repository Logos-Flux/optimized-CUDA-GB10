# optimized-CUDA-GB10

Optimized CUDA kernels for the NVIDIA GB10 Blackwell GPU (DGX Spark, sm_121).

## Performance

**Vectorized RMSNorm** — 2.59x average speedup over PyTorch baseline (BF16, GB10):

| Shape | Custom | PyTorch | Speedup |
|:---|:---:|:---:|:---:|
| [1x1024x2048] | 0.034 ms | 0.051 ms | 1.51x |
| [4x1024x2048] | 0.161 ms | 0.415 ms | 2.57x |
| [2x4096x3072] | 0.537 ms | 1.583 ms | 2.95x |
| [4x4096x3072] | 1.061 ms | 3.187 ms | 3.00x |

## Kernels

| Kernel | Vectorization | Avg Speedup | dtypes |
|--------|---------------|:-----------:|--------|
| **RMSNorm** | `__nv_bfloat162` / `__half2` / `float4` | **2.59x** | bf16, fp16, fp32 |
| **GELU** | `__nv_bfloat162` / `__half2` / `float4` | ~1.0x | bf16, fp16, fp32 |

## Quick start

```bash
# Create venv with CUDA 13 PyTorch
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install ninja pytest numpy

# Build and install kernels
TORCH_CUDA_ARCH_LIST="12.1a" .venv/bin/pip install -e . --no-build-isolation

# Run tests
.venv/bin/pytest tests/ -v

# Run benchmarks
.venv/bin/python benchmarks/benchmark_rmsnorm.py
.venv/bin/python benchmarks/benchmark_gelu.py
```

## HuggingFace Kernel Hub

The RMSNorm kernel is published on the Hub: [logos-flux/gb10-rmsnorm](https://huggingface.co/logos-flux/gb10-rmsnorm)

```python
from kernels import get_kernel

kernel = get_kernel("logos-flux/gb10-rmsnorm")
kernel.rmsnorm(out, input, weight, 1e-6)
```

## Docker

```bash
docker compose up -d --build
docker exec -it optimized-cuda-gb10 python verify_setup.py
docker exec -it optimized-cuda-gb10 pytest tests/ -v
```

## Project structure

```
├── kernels/                    # CUDA kernel sources (.cu)
│   ├── rmsnorm.cu              # Vectorized RMSNorm (bf16/fp16/fp32)
│   └── gelu.cu                 # Vectorized GELU approximate
├── torch_ext/                  # PyTorch C++ bindings + Python API
│   ├── torch_binding.cpp
│   └── gb10_kernels/__init__.py
├── benchmarks/                 # Micro-benchmarks
│   ├── benchmark_rmsnorm.py
│   └── benchmark_gelu.py
├── tests/                      # Correctness tests (21 tests, 3 dtypes)
│   └── test_kernels.py
├── hf-kernel/                  # HuggingFace Kernel Hub layout
│   ├── build.toml
│   ├── kernel_src/rmsnorm.cu
│   └── torch-ext/
├── setup.py
├── Dockerfile
└── docker-compose.yml
```

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (DGX Spark) |
| Compute Capability | 12.1 (sm_121, Blackwell) |
| Memory | 128 GB unified LPDDR5X |
| SMs | 48 |
| CUDA | 13.0 |
| BF16 Tensor Core | 92 TFLOPS (measured) |
| Memory Bandwidth | 218 GB/s (measured) |

## GB10 notes

- PyTorch's sm_121 warning is safe to ignore (sm_120/sm_121 are binary compatible)
- Do not install `flash-attn` — use PyTorch native SDPA instead
- Default `pip install torch` gives CPU-only on ARM64 — use the `cu130` index URL
- Set `TORCH_CUDA_ARCH_LIST="12.1a"` for kernel compilation

## License

Apache 2.0
