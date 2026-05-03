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

## Quick start (Docker — recommended)

```bash
DOCKER_BUILDKIT=1 docker build -t gb10-kernels:latest .
docker run --rm --gpus all gb10-kernels:latest python /workspace/verify_setup.py
docker run --rm --gpus all gb10-kernels:latest pytest /workspace/tests/ -v
docker run --rm --gpus all gb10-kernels:latest \
    python /workspace/benchmarks/benchmark_rmsnorm.py
```

Or via compose for an interactive dev shell with the source bind-mounted:

```bash
docker compose up -d --build
docker exec -it optimized-cuda-gb10 python verify_setup.py
docker exec -it optimized-cuda-gb10 pytest tests/ -v
```

## Why NGC?

The default `Dockerfile` is built on `nvcr.io/nvidia/pytorch:25.12-py3`, not
on `nvidia/cuda:13.0.0-devel` plus `pip install torch`. That choice exists
because we tried the pip path first and ran into two blocking issues on real
GB10 hardware:

1. **`libcudart.so.12` missing.** The official PyTorch `cu130` wheel still
   links against CUDA 12 system libraries. On a CUDA-13-only system (DGX
   Spark, the `nvidia/cuda:13.x-devel` image) `import torch` fails before
   you get anywhere.
2. **`CUBLAS_STATUS_INVALID_VALUE` on BF16 matmul.** That same wheel's PTX
   tops out at compute capability 12.0; GB10 is CC 12.1. Anything that
   touches the Tensor Cores in BF16 or FP16 errors out.

NGC PyTorch is built by NVIDIA against CUDA 13 with CC 12.1 PTX, so both
problems disappear. Run `verify_setup.py` to confirm — it does an explicit
BF16 matmul as the canary.

## vLLM with the GB10 RMSNorm patch

`Dockerfile.vllm` builds vLLM from source against the NGC PyTorch and wires
the GB10 RMSNorm kernel in via a `sitecustomize.py` hook.

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.vllm \
    --build-arg VLLM_REF=main \
    -t gb10-vllm:latest .
docker run --rm --gpus all -w /workspace gb10-vllm:latest \
    python -c "import vllm; from vllm import _C; print(vllm.__version__)"
```

A few things have to be done in a specific way; if you fork this and change
them you will probably break something:

- **Build vLLM from source, not from the wheel.** The published vLLM wheel
  is linked against CUDA 12 PyTorch and won't load.
- **Use `pip install --no-build-isolation --no-deps -e .`.** Without
  `--no-build-isolation` pip creates an isolated build env and installs a
  fresh torch wheel into it (losing NGC PyTorch). Without `--no-deps` pip's
  resolver pulls a CPU-only torch wheel into the runtime env. We install
  vLLM's other deps separately from `requirements/common.txt` with the
  torch lines stripped out.
- **`WORKDIR` must not be the vLLM source tree.** Python prepends the CWD
  to `sys.path` and shadows the installed `vllm` package, so it imports the
  source directory (which has no compiled `_C.abi3.so`). The image sets
  `WORKDIR /workspace`; if you override with `-w` on `docker run`, point it
  anywhere outside `/opt/vllm`.
- **A/B benchmarking against stock vLLM:** set `GB10_DISABLE_PATCH=1` to
  run the same image without the RMSNorm patch. See
  `docker-compose.kernel-test.yml` for a stock-vs-custom side-by-side.

## Building without Docker

This requires CUDA 13 and a PyTorch built with CC 12.1 PTX support
already installed on the host (i.e. NGC PyTorch unpacked into a venv, or
NVIDIA's nightly cu130 once CC 12.1 lands in the wheel). The pip
`--index-url …/whl/cu130` path does **not** currently meet this bar; see
"Why NGC?" above.

```bash
TORCH_CUDA_ARCH_LIST="12.1a" pip install -e . --no-build-isolation
pytest tests/ -v
python benchmarks/benchmark_rmsnorm.py
```

## HuggingFace Kernel Hub

The RMSNorm kernel is published on the Hub:
[logos-flux/gb10-rmsnorm](https://huggingface.co/logos-flux/gb10-rmsnorm)

```python
from kernels import get_kernel

kernel = get_kernel("logos-flux/gb10-rmsnorm")
kernel.rmsnorm(out, input, weight, 1e-6)
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
├── tests/                      # Correctness tests
├── hf-kernel/                  # HuggingFace Kernel Hub layout
├── verify_setup.py             # CC 12.1 / BF16 matmul canary
├── gb10_vllm_patch.py          # vLLM RMSNorm.forward_cuda hook
├── setup.py
├── Dockerfile                  # NGC PyTorch base, kernels only
├── Dockerfile.vllm             # NGC + vLLM from source + patch
├── docker-compose.yml          # dev shell
└── docker-compose.kernel-test.yml  # stock vs patched A/B
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
| NVFP4 / MXFP4 | Native Tensor Core support (sm_121, CUDA 13) |
| Memory Bandwidth | 218 GB/s (measured) |

## GB10 notes

- PyTorch's sm_121 warning is safe to ignore (sm_120/sm_121 are binary compatible)
- Do not install `flash-attn` — use PyTorch native SDPA instead
- Set `TORCH_CUDA_ARCH_LIST="12.1a"` for kernel compilation
- **NVFP4 / MXFP4 needs NGC PyTorch too.** GB10's Tensor Cores expose
  NVFP4 (E2M1 + per-block FP8 scale) and MXFP4 natively under CUDA 13,
  but the upstream `cu130` PyTorch wheel doesn't surface the new dtypes
  yet — same root cause as "Why NGC?" above, different downstream effect.
  If you want to run NVFP4-quantized models on GB10, stay on the NGC base.

## License

Apache 2.0
