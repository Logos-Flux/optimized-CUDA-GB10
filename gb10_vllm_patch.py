"""
Auto-patch module: patches vLLM's RMSNorm.forward_cuda with GB10 custom kernel.

This module is imported via PYTHONPATH + sitecustomize to ensure
the patch is applied in all vLLM worker processes.

To use: set PYTHONPATH to include the directory containing this file,
and rename or symlink as sitecustomize.py.

Set GB10_DISABLE_PATCH=1 to skip patching (useful for A/B benchmarking
the same image against stock vLLM).
"""
import logging
import os

logger = logging.getLogger("gb10-patch")


def _apply_patch() -> None:
    import torch  # noqa: F401
    from gb10_kernels import rmsnorm as gb10_rmsnorm
    from vllm.model_executor.layers.layernorm import RMSNorm

    def _patched_forward_cuda(self, x, residual=None):
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        if residual is not None:
            x = x + residual
            residual = x
            out = gb10_rmsnorm(x.contiguous(), self.weight.data, self.variance_epsilon)
            return out, residual
        out = gb10_rmsnorm(x.contiguous(), self.weight.data, self.variance_epsilon)
        return out

    RMSNorm.forward_cuda = _patched_forward_cuda
    logger.info("Patched vLLM RMSNorm.forward_cuda with GB10 custom kernel")


if os.environ.get("GB10_DISABLE_PATCH") == "1":
    logger.info("GB10_DISABLE_PATCH=1 — using stock vLLM RMSNorm")
else:
    try:
        _apply_patch()
    except Exception as e:
        logger.warning(f"Failed to apply GB10 RMSNorm patch: {e}")
