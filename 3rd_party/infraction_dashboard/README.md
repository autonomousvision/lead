# Infraction dashboard

A standalone Flask viewer for the driving infractions of a CARLA evaluation run.
It reads the evaluation outputs off disk and does not import `lead`.

## Run it

```bash
python 3rd_party/infraction_dashboard/app.py     # or: scripts/hotkeys/51_start_dashboard.sh
```

Then open `http://localhost:5000`. Use `--port` / `--host` to change where it
listens.

## Usage

1. **Click "Load Routes"** to scan the default output directory (`outputs/local_evaluation/`).
   To read a different one, type its path into the header input first.
1. **Select a route** in the sidebar to see its infractions.
1. **Click an infraction** to jump to that timestamp in the video.

Video shortcuts: `Space` play/pause, `←`/`→` seek 1 s, `Shift + ←`/`→` seek 5 s.

## Expected data

One directory per route, each holding `infractions.json` plus the `_debug.mp4`,
`_demo.mp4` or `_grid.mp4` videos written by the evaluation run.
