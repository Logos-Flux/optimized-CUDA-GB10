# Contributing to optimized-CUDA-GB10

Thanks for your interest in contributing. This project ships CUDA kernels
that are highly hardware-specific (NVIDIA GB10, sm_121), so a few notes
up front before you sink time into a change.

## Hardware requirements

You need access to a GB10 (or sm_121-binary-compatible) GPU to run the
test suite or benchmarks. Without one, you can still:

- Read and review code
- Propose API or doc changes
- Contribute build / packaging / CI fixes that don't require a GPU run

If you have an sm_120 consumer Blackwell card, kernels will load and run
(sm_120 / sm_121 are binary compatible), but performance numbers won't
match the README's GB10 measurements.

## Development setup

Docker is the recommended path because it pins the NGC PyTorch base
image — see [README.md → "Why NGC?"](./README.md#why-ngc) for why this
matters.

```bash
DOCKER_BUILDKIT=1 docker build -t gb10-kernels:latest .
docker run --rm --gpus all gb10-kernels:latest python /workspace/verify_setup.py
docker run --rm --gpus all gb10-kernels:latest pytest /workspace/tests/ -v
```

Or for an interactive dev shell with the source bind-mounted:

```bash
docker compose up -d --build
docker exec -it optimized-cuda-gb10 bash
```

## Workflow

1. Fork and create a topic branch off `main`.
2. Make your change. Keep commits focused — one logical change per
   commit.
3. Run the checks below.
4. Open a pull request against `main`. Fill out the PR template.

## Checks before opening a PR

```bash
# Lint + format
ruff check .
ruff format --check .   # or: black --check .

# Tests (requires GB10)
pytest tests/ -v

# Sanity check
python verify_setup.py
```

If your change is performance-relevant, include benchmark numbers
(before / after, same shape set) in the PR description. Use
`benchmarks/benchmark_rmsnorm.py` as a reference shape table.

## Code style

- **Python:** ruff + black, line length 100. Configuration is in
  `pyproject.toml`.
- **CUDA / C++:** match the surrounding file. No project-wide formatter
  is enforced for `.cu` / `.cpp` (yet); keep diffs minimal.
- **Imports:** stdlib first, third-party second, local last. Ruff's `I`
  rule enforces this.

## Reporting bugs / requesting features

Use the GitHub issue templates. For bugs, include `nvidia-smi` output,
`python -c "import torch; print(torch.__version__, torch.version.cuda)"`,
and ideally `verify_setup.py` output.

## License

By contributing, you agree your contributions are licensed under
Apache-2.0 (the project's license).
