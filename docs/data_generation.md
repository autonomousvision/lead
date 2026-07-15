# Data collection

The expert drives CARLA routes and writes a py123d dataset to `PY123D_DATA_ROOT`. Start CARLA and run:

```bash
python -m lead --expert --routes lead/src/lead/routes/data_routes/lead/Accident/route_001761.xml
```

Route files live under `src/lead/routes/data_routes/`, grouped by scenario type (e.g.
`lead/Accident/`, `lead/noScenarios/`); point `--routes` at any single file or a
directory to collect over all routes inside it.

Collection throughput depends on the GPU driving CARLA's rendering: around 10 Hz on a
GTX 1080 Ti with 2 CPUs, up to 70-80 Hz on an RTX 5090.
