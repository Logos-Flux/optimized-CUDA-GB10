import torch

from ._ops import ops


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    RMSNorm: x * rsqrt(mean(x^2) + eps) * weight

    Vectorized CUDA kernel optimized for NVIDIA GB10 Blackwell (sm_121).
    2.59x average speedup over PyTorch baseline.

    Args:
        input: Input tensor [..., hidden_size]. Supports float32, float16, bfloat16.
        weight: Scale tensor [hidden_size].
        eps: Epsilon for numerical stability. Default: 1e-6.
        out: Optional pre-allocated output tensor.

    Returns:
        Normalized tensor, same shape and dtype as input.
    """
    if out is None:
        out = torch.empty_like(input)
    ops.rmsnorm(out, input.contiguous(), weight.contiguous(), eps)
    return out


__all__ = ["rmsnorm"]
