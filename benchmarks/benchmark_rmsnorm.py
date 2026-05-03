#!/usr/bin/env python3
"""
Micro-benchmark: RMSNorm custom kernel vs PyTorch baseline on GB10.

Usage:
    python benchmarks/benchmark_rmsnorm.py
"""

import time

import torch
from gb10_kernels import rmsnorm


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference PyTorch implementation of RMSNorm."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def benchmark_fn(func, args, warmup: int = 20, iterations: int = 100) -> tuple[float, float]:
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
    print("RMSNorm Micro-Benchmark: Custom Kernel vs PyTorch Baseline")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print()

    configs = [
        (1, 1024, 2048),
        (2, 1024, 2048),
        (4, 1024, 2048),
        (1, 4096, 2048),
        (2, 4096, 3072),
        (1, 8192, 2048),
        (4, 4096, 3072),
    ]

    dtype = torch.bfloat16

    print(f"{'Config':<25} {'Custom (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    total_speedup = 0.0
    n = 0

    for batch, seq, hidden in configs:
        x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
        weight = torch.ones(hidden, dtype=dtype, device="cuda")

        custom_avg, _ = benchmark_fn(rmsnorm, (x, weight, 1e-6))
        pytorch_avg, _ = benchmark_fn(pytorch_rmsnorm, (x, weight, 1e-6))

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
    weight = torch.ones(2048, dtype=dtype, device="cuda")

    custom_out = rmsnorm(x, weight, 1e-6)
    pytorch_out = pytorch_rmsnorm(x, weight, 1e-6)

    max_diff = (custom_out - pytorch_out).abs().max().item()
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Correctness: {'PASS' if max_diff < 0.05 else 'FAIL'}")
    print()

    # Bandwidth analysis
    print("Memory Bandwidth Analysis:")
    batch, seq, hidden = 4, 4096, 3072
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.ones(hidden, dtype=dtype, device="cuda")

    bytes_per_elem = 2  # bfloat16
    total_bytes = (2 * batch * seq * hidden + hidden) * bytes_per_elem
    custom_avg, _ = benchmark_fn(rmsnorm, (x, weight, 1e-6))

    bandwidth_gbps = (total_bytes / 1e9) / (custom_avg / 1000)
    print(f"  Total data moved: {total_bytes / 1e6:.2f} MB")
    print(f"  Achieved bandwidth: {bandwidth_gbps:.1f} GB/s")
    # GB10 theoretical bandwidth — unified memory, not HBM
    print("  (GB10 has unified memory — bandwidth profile differs from HBM GPUs)")


if __name__ == "__main__":
    main()
