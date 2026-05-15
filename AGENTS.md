# AGENTS.md

## Project Overview

This is a Python project for a SLURM job launcher. The runner reads partition YAML files, expands parameter sweeps, generates sbatch files, submits jobs, and records submitted job IDs.

Primary reference documents:

- `SPEC.md` — desired runner behavior and feature requirements.
- `partitions/*.yaml` — runtime partition/project configuration.

## Development Guidelines

- Use Python 3.10+ unless the project later specifies otherwise.
- Use `uv` for Python environment, dependency, packaging, and command execution.
- Prefer small, focused modules over one large script.
- Keep cluster/partition-specific data in YAML files, not Python code.
- Do not hard-code any cluster/partition configuration in code. Everything should be configurable via YAML.
- Keep CLI behavior deterministic and reproducible.

## Code Style

- Follow PEP 8.
- Use type hints for public functions and non-trivial internal functions.
- Prefer `pathlib.Path` over raw string path manipulation.
- Prefer dataclasses or typed models for loaded configuration.
- Validate user input early and fail with clear error messages.
- Avoid broad `except Exception` blocks unless re-raising with useful context.

## Testing

- Add tests for new behavior when implementing features.
- Prefer `pytest`.
- Unit-test pure logic separately from remote SSH/SLURM submission.
- Mock subprocess calls for `git`, `ssh`, `rsync`, and `sbatch`.
- Important behavior to test:
  - YAML partition loading and validation.
  - Hydra-style and positional sweep expansion.
  - Control parameter parsing.
  - Multi-GPU resource calculation.
  - `task_mode: per_gpu` and `task_mode: per_node`.
  - sbatch script generation.
  - job ID parsing and `job_id.txt` writing.
  - remote version synchronization decision logic.

## Commands

Use `uv` for project commands. When project tooling exists, prefer:

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy .
```

For dependency management, prefer:

```bash
uv add <package>
uv add --dev <package>
uv sync
```

If a tool is not configured yet, add it through `uv` when appropriate or ask first.

## Dependencies

- Keep dependencies minimal.
- Manage dependencies with `uv` and `pyproject.toml`.
- Use `pyyaml` or a comparable YAML parser for partition files.
- Do not manually edit lockfile content; let `uv` update it.

## Git / Remote Safety

- Do not submit jobs as part of tests.
- Do not run destructive remote commands without explicit user approval.
- Any remote synchronization code should support dry-run or mockable execution.
- Before submission, the runner must ensure the remote launcher version matches the local version as specified in `SPEC.md`.

## Documentation

- Update `SPEC.md` when requirements change.
- Keep partition YAML examples and schema expectations aligned with code.
- Document any new CLI options in the README when one is added.
