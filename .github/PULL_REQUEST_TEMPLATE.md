## Summary

<!-- What does this PR do, and why? Keep it tight. -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (API or behavior)
- [ ] Build / packaging / CI
- [ ] Docs only

## Checks

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` (or `black --check .`) passes
- [ ] `pytest tests/ -v` passes on a GB10 (or sm_121-compatible) GPU
- [ ] `python verify_setup.py` still passes
- [ ] CHANGELOG.md updated under `## [Unreleased]` if user-visible
- [ ] No new internal references, secrets, or PII

## Performance (if perf-relevant)

<!-- Run benchmarks/benchmark_rmsnorm.py (or equivalent) before and
     after. Paste the relevant rows. State the GPU + dtype + shape set. -->

| Shape | Before | After | Δ |
|---|---|---|---|
| | | | |

## Notes for the reviewer

<!-- Anything tricky? Anything you weren't sure about? -->
