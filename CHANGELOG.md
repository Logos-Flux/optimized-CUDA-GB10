# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-03

### Added
- Vectorized **RMSNorm** CUDA kernel for NVIDIA GB10 Blackwell (sm_121).
  Uses `__nv_bfloat162` / `__half2` / `float4` packed loads. Average
  2.59x speedup over PyTorch baseline on BF16 across the measured shape
  set (1.51x – 3.00x).
- Vectorized **GELU** CUDA kernel (approximate, tanh form). bf16 / fp16 / fp32.
- PyTorch C++ extension (`gb10_kernels._C`) and Python wrapper module
  exposing `rmsnorm(input, weight, eps, out=None)` and `gelu(input, out=None)`.
- HuggingFace Kernel Hub layout under `hf-kernel/` for publishing
  `logos-flux/gb10-rmsnorm`. Includes a local builder script for the
  cu130 + aarch64 case that the upstream Nix `kernel-builder` does not
  yet handle.
- vLLM integration via `gb10_vllm_patch.py` + `sitecustomize.py` hook in
  `Dockerfile.vllm`. `GB10_DISABLE_PATCH=1` reverts to stock vLLM RMSNorm
  for A/B benchmarking.
- `verify_setup.py` — CC 12.1 + BF16 matmul canary check.
- Two Dockerfiles based on `nvcr.io/nvidia/pytorch:25.12-py3` (NGC) — one
  for the kernels alone, one bundling vLLM built from source.
- Apache-2.0 license.

### Notes
- Requires CUDA 13 and a PyTorch build that emits CC 12.1 PTX. The
  upstream `cu130` PyTorch wheel does not currently meet this bar; see
  README "Why NGC?" for details.
- Targets sm_121 specifically. sm_120 (consumer Blackwell) is binary
  compatible at runtime.

[Unreleased]: https://github.com/Logos-Flux/optimized-CUDA-GB10/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Logos-Flux/optimized-CUDA-GB10/releases/tag/v0.1.0
