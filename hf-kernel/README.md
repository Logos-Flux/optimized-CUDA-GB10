---
tags:
  - kernel
  - cuda
  - rmsnorm
  - blackwell
  - gb10
  - sm_121
library_name: kernels
---

# GB10 RMSNorm — Vectorized CUDA Kernel for Blackwell (sm_121)

**The first sm_121 (compute capability 12.1) kernel on the HuggingFace Kernel Hub.**

Optimized RMSNorm implementation for the NVIDIA GB10 Blackwell GPU (DGX Spark). Uses vectorized memory access (`__nv_bfloat162`, `__half2`, `float4`) for 2-4x element throughput per load.

## Performance

**2.59x average speedup** over PyTorch baseline on NVIDIA GB10 (128GB unified VRAM):

| Shape | Custom (ms) | PyTorch (ms) | Speedup |
|:---|:---:|:---:|:---:|
| [1x1024x2048] | 0.034 | 0.051 | **1.51x** |
| [2x1024x2048] | 0.063 | 0.154 | **2.44x** |
| [4x1024x2048] | 0.161 | 0.415 | **2.57x** |
| [1x4096x2048] | 0.158 | 0.441 | **2.78x** |
| [2x4096x3072] | 0.537 | 1.583 | **2.95x** |
| [1x8192x2048] | 0.356 | 1.013 | **2.84x** |
| [4x4096x3072] | 1.061 | 3.187 | **3.00x** |

Achieved bandwidth: **185.4 GB/s** (GB10 unified memory).

## Quick Start

```python
from kernels import get_kernel
import torch

kernel = get_kernel("logos-flux/gb10-rmsnorm")

x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16, device="cuda")
weight = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
out = torch.empty_like(x)

kernel.rmsnorm(out, x, weight, 1e-6)
```

## Supported Data Types

- **bfloat16** — vectorized via `__nv_bfloat162` (2 elements per load)
- **float16** — vectorized via `__half2` (2 elements per load)
- **float32** — vectorized via `float4` (4 elements per load)

Falls back to scalar kernel for hidden sizes < 64 or odd dimensions.

## API

```python
rmsnorm(input, weight, eps=1e-6, out=None) -> torch.Tensor
```

**Parameters:**
- `input` — Input tensor `[..., hidden_size]`
- `weight` — Scale tensor `[hidden_size]`
- `eps` — Epsilon for numerical stability (default: `1e-6`)
- `out` — Optional pre-allocated output tensor

**Returns:** Normalized tensor, same shape and dtype as input.

## Integration with Transformers

```python
from kernels import get_kernel
from transformers import AutoModelForCausalLM
import torch

kernel = get_kernel("logos-flux/gb10-rmsnorm")

def patch_rmsnorm(model):
    for name, module in model.named_modules():
        if 'RMSNorm' in type(module).__name__:
            eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)
            def make_forward(mod, epsilon):
                def forward(x):
                    out = torch.empty_like(x)
                    kernel.rmsnorm(out, x.contiguous(), mod.weight.contiguous(), epsilon)
                    return out
                return forward
            module.forward = make_forward(module, eps)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
model.to("cuda")
patch_rmsnorm(model)
```

## Integration with Diffusers

```python
from kernels import get_kernel
from diffusers import LTXPipeline
import torch

kernel = get_kernel("logos-flux/gb10-rmsnorm")

def patch_rmsnorm(model):
    for name, module in model.named_modules():
        if type(module).__name__ == 'RMSNorm':
            eps = getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None
            if has_weight:
                def make_forward(mod, epsilon):
                    def forward(x):
                        out = torch.empty_like(x)
                        kernel.rmsnorm(out, x.contiguous(), mod.weight.contiguous(), epsilon)
                        return out
                    return forward
                module.forward = make_forward(module, eps)
            else:
                def make_forward(epsilon):
                    def forward(x):
                        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
                        out = torch.empty_like(x)
                        kernel.rmsnorm(out, x.contiguous(), w, epsilon)
                        return out
                    return forward
                module.forward = make_forward(eps)

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")
patch_rmsnorm(pipe.transformer)
```

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (DGX Spark) |
| Compute Capability | sm_121 (Blackwell) |
| VRAM | 128 GB unified |
| CUDA | 13.0 |
| PyTorch | 2.10+ |

## Build Info

Built locally on NVIDIA GB10 (aarch64-linux, CUDA 13, PyTorch 2.10).
The kernel-builder Nix pipeline does not yet support cu130+aarch64, so this was compiled using PyTorch's JIT extension loader with `--gencode=arch=compute_120,code=sm_121`.

## Source

[github.com/CyberBrown/optimized-CUDA-GB10](https://github.com/CyberBrown/optimized-CUDA-GB10)
