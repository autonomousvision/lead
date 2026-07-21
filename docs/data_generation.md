# Data collection

## Local collection

The expert drives CARLA routes and writes a py123d dataset to `PY123D_DATA_ROOT`. Start CARLA and run:

```bash
python -m lead --expert --routes lead/src/lead/routes/data_routes/lead/Accident/route_001761.xml
```

## Parallel collection on SLURM

For collecting many routes in parallel on a SLURM cluster:

```bash
python scripts/slurm/collect_data.py
```

Every route XML becomes its own SLURM job with a private CARLA instance. The launcher
keeps a fixed pool of jobs running and retries failed routes; pool size, retry limit,
and SLURM resources are set in `.env` (see `.env.example`). A route counts as finished
once its result file shows a completed run with a nonzero route score, so re-running
the launcher only collects what is missing.

## Common issues

**Corrupted maps after parallel collection.** py123d maps are shared: one map per town,
used by all logs. A writer converts a missing map on first use, so parallel jobs in the
same town can race and corrupt the shared map files. Convert all maps once before
starting parallel jobs:

```bash
python scripts/common/convert_py123d_maps.py
```

Already-converted towns are skipped; `collect_data.py` runs this automatically. To
recover a corrupted map, delete its `.arrow` file under `<PY123D_DATA_ROOT>/maps/` and
re-run the conversion.
