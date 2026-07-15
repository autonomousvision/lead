# Setup details

See the [README](../README.md#setup-for-development) for the install walkthrough. This page covers what the install actually sets up.

## `setup.py`

Runs on `uv sync`:

- Bootstraps `.env` from `.env.example` (fills in missing keys on later installs, never touches existing values).
- Adds `scripts/bin` and `scripts/common` to `PATH` via a conda/micromamba `activate.d` hook (or a line in `bin/activate` for a plain venv) — this only runs on `conda activate lead` / `micromamba activate lead`, so activate the env in each new shell before using repo scripts.

## `.env`

Gitignored, machine-local config: simulator/data paths, SLURM job settings, WandB logging cadence, and `LEAD_RUNTIME_TYPE_CHECKING` (beartype/jaxtyping toggle — off by default, on for debugging/tests). Re-read on every access, so edits apply without a restart. `.env.example` is the checked-in template.
