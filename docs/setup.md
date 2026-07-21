# Setup details

## Run tests and lints

```bash
pre-commit install

# Unit tests with dynamic type checking enabled
LEAD_RUNTIME_TYPE_CHECKING=true pytest tests/unittests

# Run pre-commit with static type checking
pre-commit run --all-files
```

The end-to-end tests assert the invariants of a collected dataset — collect at
least one route first (see [data collection](data_generation.md)), then point
them at the dataset root. Without it they skip:

```bash
LEAD_TEST_DATA_ROOT=<data root, e.g. $PY123D_DATA_ROOT> pytest tests/e2e_tests
```

## `setup.py`

Runs on `uv sync`:

- Bootstraps `.env` from `.env.example` (fills in missing keys on later installs, never touches existing values).
- Adds `scripts/cli` and `scripts/common` to `PATH` via a conda/micromamba `activate.d` hook (or a line in `bin/activate` for a plain venv). This only runs on `conda activate lead` / `micromamba activate lead`, so activate the env in each new shell before using repo scripts.

## `.env`

Gitignored, machine-local config: simulator/data paths, SLURM job settings, WandB logging cadence, and `LEAD_RUNTIME_TYPE_CHECKING`.

Some values are re-read on every access, so edits of those values apply without a restart.
