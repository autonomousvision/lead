# Evaluation

Start CARLA and run:

```bash
# Standard eval
python -m lead \
    --checkpoint <ckpt> \
    --routes src/lead/routes/benchmark_routes/Town13/0.xml

# Bench2Drive
python -m lead --checkpoint <ckpt> --routes <route> --bench2drive

# Fail2Drive
python -m lead --checkpoint <ckpt> --routes <route> --fail2drive
```

`--expert` runs the expert agent instead of a checkpoint (used for data generation, see
[Data collection](data_generation.md)), and requires no `--checkpoint`. `--bench2drive` and
`--fail2drive` each select a different vendored leaderboard/scenario-runner under
`3rd_party/leaderboard/` (and, for Fail2Drive, a separate CARLA build) instead of the standard
one; omit both for the standard benchmark. Routes live under `src/lead/routes/benchmark_routes/`
(`Town13/`, `bench2drive/`, `longest6/`, `fail2drive/`).

Other useful flags: `--repetitions` (repeat each route), `--resume` (continue into an existing
output directory instead of clearing it), `--output-dir` (override the default output location),
`--port`/`--traffic-manager-port` (for running multiple evaluations against different CARLA
instances on one machine). Run `python -m lead --help` for the full list.

## Output

Each run writes to `outputs/local_evaluation/<route_id>/` by default (`outputs/expert_evaluation/`
for `--expert`), one directory per evaluation:

- `checkpoint_endpoint.json` — the leaderboard's own result file: per-route driving score,
  route completion, infraction penalty, and status.
- `infractions.json` — per-infraction log written by `InfractionRecorder`
  (`lead.evaluation.recorder.infraction_recorder`): one entry per detected infraction with the
  step, frame, criterion name, message, and distance travelled.
- `<route_id>_debug.mp4` / `<route_id>_demo.mp4` / `<route_id>_grid.mp4` — optional recorded
  videos, enabled via `EvaluationConfig.produce_debug_video` / `produce_demo_video` /
  `produce_grid_video` (`lead.config.evaluation.evaluation_config`); off by default except
  `produce_debug_video`, which follows `debug_mode`.

## Infraction dashboard

`3rd_party/infraction_dashboard/` is a standalone Flask viewer for these outputs — it reads
`infractions.json` and the recorded videos off disk and does not import `lead`.

```bash
python 3rd_party/infraction_dashboard/app.py   # then open http://localhost:5000
```

It defaults to `outputs/local_evaluation/`; point it elsewhere with the path field in its header,
or override `--host`/`--port` on the CLI. Load routes in the sidebar, select one to see its
infractions, and click an infraction to jump to that timestamp in the video. See
[3rd_party/infraction_dashboard/README.md](../3rd_party/infraction_dashboard/README.md) for
details.
