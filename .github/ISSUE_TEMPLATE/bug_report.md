---
name: Bug report
about: Report a correctness, build, or runtime bug
title: "[bug] "
labels: bug
assignees: ''
---

## Summary

<!-- One or two sentences describing what's wrong. -->

## Repro

<!-- Minimal code or commands that reproduce the issue. -->

```python

```

## Expected vs. actual

- **Expected:**
- **Actual:**

## Environment

Please paste the output of:

```bash
nvidia-smi | head -20
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cxx11', torch._C._GLIBCXX_USE_CXX11_ABI)"
python verify_setup.py 2>&1 | tail -30
```

| Field | Value |
|---|---|
| GPU model | <!-- e.g. NVIDIA GB10 (DGX Spark) / RTX 50xx --> |
| Compute capability | <!-- e.g. 12.1 / 12.0 --> |
| CUDA version | <!-- e.g. 13.0 --> |
| PyTorch version | <!-- e.g. 2.10.0a0+nv251212 --> |
| Build mode | <!-- Docker (NGC) / pip / from source --> |
| OS / arch | <!-- e.g. Ubuntu 24.04 / aarch64 --> |
| Project version / commit | <!-- v0.1.0 / abc1234 --> |

## Additional context

<!-- Logs, stack traces, related issues, anything else helpful. -->
