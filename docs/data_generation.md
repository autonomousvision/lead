# Data collection

## Local collection

The expert drives CARLA routes and writes a py123d dataset to `PY123D_DATA_ROOT`. Start CARLA and run:

```console
user@host:~/lead$ python -m lead --expert --routes src/lead/routes/data_routes/lead/Accident/route_001761.xml
```

## Parallel collection on SLURM

For collecting many routes in parallel on a SLURM cluster:

```console
user@host:~/lead$ python scripts/slurm/collect_data.py
```

Every route XML becomes its own SLURM job with a private CARLA instance. The launcher
keeps a fixed pool of jobs running and retries failed routes; pool size, retry limit,
and SLURM resources are set in `.env` (see `.env.example`). A route counts as finished
once its result file shows a completed run with a nonzero route score, so re-running
the launcher only collects what is missing.

## Change the sensor rig

The rig is `SensorRigConfig` in
[sensor_rig_config.py](../src/lead/config/expert/sensor_rig_config.py): two lidars, a
list of cameras, a list of radars, edited in place. It both spawns the CARLA sensors and
becomes the calibration written into the log.

<details>
<summary>Mounting a different camera rig</summary>

A single front camera instead of the six-camera surround rig:

```python
cameras: list[CameraSpec] = [
    {  # front
        "pos": [0.25, 0.0, 2.25],
        "rot": [0.0, 0.0, 0.0],
        "width": 1024,
        "height": 512,
        "fov": 90,
    },
]
```

Camera `i` is stored under the ID that `CAMERA_ID_BY_LEAD_INDEX` in
[py123d_log_api.py](../src/lead/api/py123d_log_api.py) maps it to; more than the six
mapped cameras (or four radars) needs an entry each. All cameras share the resolution of
`cameras[0]`, and the rig is always two lidars. Depth, instance and perturbated cameras
are derived from the RGB list; a policy's camera selection
(`policy.transfuser.camera.input_cameras`) is not. Scalar knobs also work via the
`LEAD_CONFIG` dotlist, e.g. `export LEAD_CONFIG="expert.sensor_rig.use_radars=false"`.

Training reads `<PY123D_DATA_ROOT>/config.yaml` as its expert section, so store the
config with the data. `scripts/slurm/collect_data.py` does this; after a local run:

```python
import yaml

from lead.config import load_lead_config, yaml_filtered

expert_config = yaml_filtered(load_lead_config().expert.to_dict())
with open("data/lead/123D/config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(expert_config, f, sort_keys=False)
```

</details>

## Add a modality offline

A log is one Arrow file per modality stream plus `sync.arrow`, which holds one row per
tick with the row index each stream sits at. To add a modality after collection
(reasoning traces, captions, auto-labels), write `custom.<id>.arrow` next to the
others; closing the writer rebuilds the sync table from every `*.arrow` in the
directory. It reads back like any other stream:
`scene.get_custom_modality_at_iteration(0, "reasoning").data`.

<details>
<summary>Example code</summary>

```python
from pathlib import Path

import pyarrow as pa
from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter, SyncConfig
from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.datatypes import CustomModality, CustomModalityMetadata, Timestamp

logs_root = Path("data/lead/123D/logs")
log_dir = logs_root / "normal_view/Accident/Town03_Rep0_route_001783_route0"

scenes = ArrowSceneBuilder(
    logs_root=str(logs_root),
    maps_root="data/lead/123D/maps",
).get_scenes(
    SceneFilter(
        split_names=["normal_view/Accident"],
        log_names=[log_dir.name],
        future_num_iterations=0,
    ),
    ThreadPoolExecutor(),
)
log_metadata = scenes[0].get_log_metadata()

sync = pa.ipc.open_file(pa.memory_map(str(log_dir / "sync.arrow"))).read_all()
timestamps = sync.column("sync.timestamp_us").to_pylist()

writer = ArrowLogWriter(
    log_writer_config=LogWriterConfig(force_log_conversion=True),
    logs_root=logs_root,
    sensors_root=Path("data/lead/123D/sensors"),
    sync_config=SyncConfig(
        reference_column="custom.driving_meta.timestamp_us",
        direction="backward",
    ),
)
writer.reset(log_metadata)

metadata = CustomModalityMetadata(modality_id="reasoning", metadata={"model": "..."})
for time_us in timestamps:
    writer.write_async(
        CustomModality(
            data={"text": your_caption_for(time_us)},
            metadata=metadata,
            timestamp=Timestamp.from_us(time_us),
        ),
    )
writer.close()
```

Values go through msgpack, so numpy arrays are fine and numpy scalars such as `np.bool_`
are not. Timestamps must be non-decreasing and need not cover every tick.
`reference_column` has to stay the stream the log was written against, and the rebuild
overwrites `sync.arrow` in place.

</details>
