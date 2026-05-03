"""
gb10_kernels - Optimized CUDA kernels for NVIDIA GB10 Blackwell (sm_121)

Provides:
    rmsnorm(input, weight, eps=1e-6) -> Tensor
    gelu(input) -> Tensor
"""

import torch

# Import the compiled C++ extension
from gb10_kernels._C import gelu_forward, rmsnorm_forward


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    RMSNorm: x * rsqrt(mean(x^2) + eps) * weight

    Args:
        input: Input tensor [..., hidden_size]
        weight: Scale tensor [hidden_size]
        eps: Epsilon for numerical stability
        out: Optional pre-allocated output tensor

    Returns:
        Normalized tensor, same shape as input
    """
    if out is None:
        out = torch.empty_like(input)

    rmsnorm_forward(out, input.contiguous(), weight.contiguous(), eps)
    return out


def gelu(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    GELU approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        input: Input tensor (any shape)
        out: Optional pre-allocated output tensor

    Returns:
        Activated tensor, same shape as input
    """
    if out is None:
        out = torch.empty_like(input)

    gelu_forward(out, input.contiguous())
    return out
