# Repository Guidelines

## Project Overview

This repository contains `microhh_processing_chain`, a Python package for
preprocessing, running, and postprocessing MicroHH simulation workflows. The
package uses a `src/` layout and includes notebooks, utility scripts, small
sample datasets, and generated images used for workflow development and
analysis.

## Repository Layout

- `src/microhh_processing_chain/`: importable package code.
- `scripts/`: runnable workflow and postprocessing scripts.
- `scripts/dev/`: development setup helpers and companion package requirements.
- `notebooks/`: exploratory and workflow notebooks. Keep notebook outputs
  stripped unless there is a specific reason to retain them.
- `data/`: small sample NetCDF inputs used by examples and development.
- `images/`: generated or reference figures for documentation and analysis.

## Environment And Dependencies

- Python `>=3.10` is required.
- Install the package in editable mode with `pip install -e .`.
- Development extras are declared in `pyproject.toml` under `dev`.
- Some workflows depend on local companion packages listed in `README.md` and
  `scripts/dev/requirements.txt`: `ddeq`, `u-cat`, `microhh_py`, and
  `microhhpy`.
- Use `./scripts/dev/setup_repos.sh` only when the active Python environment is
  ready and companion repositories should be cloned or updated next to this
  repository.

## Coding Conventions

- Prefer existing package patterns in `src/microhh_processing_chain/` over new
  abstractions.
- Keep scientific data handling explicit. Use `xarray`, `pandas`,
  `numpy`, and `Path` APIs rather than ad hoc parsing when possible.
- Type annotations are expected for new public functions and methods.
- Preserve the existing import style: standard library imports first, third-party
  imports next, then local package imports.
- Keep comments focused on non-obvious scientific, filesystem, or HPC behavior.
- Avoid committing generated Python caches, local environment files, large
  outputs, or bulky simulation artifacts.

## Notebooks And Data

- Treat notebooks as development artifacts unless the user asks for notebook
  changes explicitly.
- If editing notebooks, strip outputs before finishing. The pre-commit
  configuration runs `nbstripout`.
- Do not add large NetCDF or simulation output files without an explicit request.
  The pre-commit hook rejects added files larger than 6000 KB.

## MicroHH And HPC Workflow Notes

- Be careful with path handling for scratch directories, case directories,
  executables, and NetCDF outputs. Prefer `pathlib.Path`.
- Do not run remote jobs, submit scheduler scripts, transfer data, or invoke SSH
  commands unless the user explicitly asks for that action.
- For MicroHH configuration details, verify parameter names and expected NetCDF
  variables against the relevant MicroHH source or documentation before changing
  behavior.
