#!/usr/bin/env python3
"""
Micro-benchmark: GELU custom kernel vs PyTorch baseline on GB10.

Usage:
    python benchmarks/benchmark_gelu.py
"""

import torch
import time
from typing import Tuple

from gb10_kernels import gelu


def benchmark_fn(func, args, warmup: int = 20, iterations: int = 100) -> Tuple[float, float]:
    """Benchmark a function, returning (avg_ms, min_ms)."""
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        func(*args)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return sum(times) / len(times), min(times)


def main():
    print("=" * 70)
    print("GELU Micro-Benchmark: Custom Kernel vs PyTorch Baseline")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    configs = [
        (1, 1024, 2048),
        (2, 1024, 2048),
        (4, 1024, 4096),
        (2, 4096, 3072),
        (4, 4096, 3072),
        (1, 8192, 4096),
    ]

    dtype = torch.bfloat16
    pytorch_gelu = torch.nn.GELU(approximate="tanh")

    print(f"{'Config':<25} {'Custom (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    total_speedup = 0.0
    n = 0

    for batch, seq, hidden in configs:
        x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")

        custom_avg, _ = benchmark_fn(gelu, (x,))
        pytorch_avg, _ = benchmark_fn(pytorch_gelu, (x,))

        speedup = pytorch_avg / custom_avg
        total_speedup += speedup
        n += 1

        config_str = f"[{batch}x{seq}x{hidden}]"
        print(f"{config_str:<25} {custom_avg:>12.3f}   {pytorch_avg:>12.3f}   {speedup:>8.2f}x")

    print("-" * 70)
    print(f"{'Average Speedup:':<55} {total_speedup / n:.2f}x")
    print()

    # Correctness check
    print("Correctness Check:")
    x = torch.randn(2, 1024, 2048, dtype=dtype, device="cuda")
    custom_out = gelu(x)
    pytorch_out = pytorch_gelu(x)

    max_diff = (custom_out - pytorch_out).abs().max().item()
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Correctness: {'PASS' if max_diff < 0.05 else 'FAIL'}")


if __name__ == "__main__":
    main()
