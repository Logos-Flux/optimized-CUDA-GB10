# Security Policy

## Supported versions

This project follows a "latest-only" support model. Only the most recent
release on `main` receives security fixes. Older tags are kept for
reproducibility but are not patched.

| Version | Supported |
| ------- | :-------: |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

If you've found a vulnerability that you believe affects users of this
project, please report it privately rather than opening a public issue.

**Email:** lf@logosflux.io
**Subject line:** `[SECURITY] optimized-CUDA-GB10`

Please include:

- A description of the issue and its impact
- Steps to reproduce (or a proof-of-concept)
- The commit SHA / version where you observed the issue
- Any suggested mitigations, if you have them

You can expect an initial response within 7 days. Once a fix is ready,
we'll coordinate a disclosure timeline with you.

## Out of scope

This project ships CUDA kernels that run inside a user's own process on
their own hardware. The threat model is small. The following are not
security issues for this repo:

- Numerical precision differences between the custom kernels and PyTorch
  reference implementations (those are correctness issues — open a
  regular bug report)
- Performance regressions (also a regular bug)
- Issues in dependencies (`torch`, NGC PyTorch base image, vLLM) — please
  report those upstream
- Issues that require local arbitrary-code-execution to exploit (you
  already had the host)

## Acknowledgements

Reporters who follow this process and want public credit will be named
in the release notes once a fix lands.
