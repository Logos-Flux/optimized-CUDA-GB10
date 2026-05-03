"""Verify CUDA + PyTorch setup on GB10.

This is the canary script for the two issues we hit when we tried to use
the official PyTorch cu130 pip wheel on DGX Spark:

  1. libcudart.so.12 missing — torch import fails outright. Fixed by
     using the NGC PyTorch image instead of pip cu130.
  2. CUBLAS_STATUS_INVALID_VALUE on BF16 matmul — the cu130 wheel's PTX
     tops out at compute capability 12.0, GB10 is 12.1. NGC PyTorch is
     built with CC 12.1 PTX. The BF16 matmul below is the smallest test
     that exercises that path; if it succeeds, the rest of your stack
     should too.

Exits non-zero on any failure so it doubles as a CI smoke test.
"""

import sys
import time

import torch


def main() -> int:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return 1

    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    print(f"Device:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:            {props.total_memory / 1e9:.1f} GB")
    print(f"Compute cap:     {cc}")
    print(f"CUDA runtime:    {torch.version.cuda}")

    # FP32 matmul — basic CUDA path. Catches Issue #1.
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        torch.mm(x, y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = (2 * 1024**3 * 100) / elapsed / 1e12
    print(f"FP32 matmul:     {tflops:.2f} TFLOPS (1024x1024 x100)")

    # BF16 matmul — Tensor Core path. Catches Issue #2 on GB10.
    # If PyTorch's PTX doesn't cover CC 12.1 this raises
    # CUBLAS_STATUS_INVALID_VALUE.
    try:
        xb = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        yb = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            torch.mm(xb, yb)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = (2 * 2048**3 * 100) / elapsed / 1e12
        print(f"BF16 matmul:     {tflops:.2f} TFLOPS (2048x2048 x100)")
    except RuntimeError as e:
        print(f"ERROR: BF16 matmul failed: {e}")
        if cc == "12.1":
            print("  This is Issue #2 — your PyTorch was not built with")
            print("  CC 12.1 PTX. Use the NGC PyTorch image (see README).")
        return 1

    # If gb10_kernels is installed, exercise it too.
    try:
        import gb10_kernels  # noqa: F401
        from gb10_kernels import rmsnorm

        x = torch.randn(2, 1024, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
        out = rmsnorm(x, w, 1e-6)
        assert out.shape == x.shape
        print("gb10_kernels:    OK (rmsnorm bf16 forward)")
    except ImportError:
        print("gb10_kernels:    not installed (skipping kernel check)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
