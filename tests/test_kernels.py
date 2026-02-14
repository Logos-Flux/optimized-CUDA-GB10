#!/usr/bin/env python3
"""
Correctness tests for GB10-optimized CUDA kernels.

Usage:
    pytest tests/test_kernels.py -v
"""

import torch
import pytest

from gb10_kernels import rmsnorm, gelu


# ============================================================================
# RMSNorm tests
# ============================================================================

def pytorch_rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [
    (2, 1024, 2048),
    (1, 4096, 3072),
    (4, 512, 1024),
])
def test_rmsnorm_correctness(dtype, shape):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(shape[-1], dtype=dtype, device="cuda")

    expected = pytorch_rmsnorm(x, weight, 1e-6)
    result = rmsnorm(x, weight, 1e-6)

    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


def test_rmsnorm_preallocated():
    x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)

    result = rmsnorm(x, weight, 1e-6, out=out)
    assert result.data_ptr() == out.data_ptr()


def test_rmsnorm_no_weight_ones():
    """Test pattern used when elementwise_affine=False (weight=ones)."""
    x = torch.randn(1, 512, 2048, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(2048, dtype=torch.bfloat16, device="cuda")

    result = rmsnorm(x, weight, 1e-6)
    expected = pytorch_rmsnorm(x, weight, 1e-6)

    torch.testing.assert_close(result, expected, rtol=1e-2, atol=5e-2)


# ============================================================================
# GELU tests
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [
    (2, 1024, 2048),
    (1, 4096, 3072),
    (4, 512, 4096),
])
def test_gelu_correctness(dtype, shape):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    pytorch_gelu = torch.nn.GELU(approximate="tanh")

    expected = pytorch_gelu(x)
    result = gelu(x)

    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


def test_gelu_preallocated():
    x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)

    result = gelu(x, out=out)
    assert result.data_ptr() == out.data_ptr()
