"""The typed sample and driving-meta contracts, checked against real logs.

The declared dtypes and shapes of :class:`TransfuserTrainingSample` are read off its own
annotations, so this test cannot drift from the contract it verifies. Runtime
type checking enforces the same thing when it is enabled; these assertions hold
either way and report the offending field by name.
"""

import dataclasses
import typing

import numpy as np
import pytest

from lead.api.driving_meta import RequiredDrivingMeta
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserTrainingSample

# Every sample decodes full images, so inspect fewer of them than the loader
# invariants do.
MAX_SAMPLES = 32


@pytest.fixture(scope="module")
def dataset(lead_config: LeadConfig):
    """The TransFuser dataset over the configured logs.

    Args:
        lead_config: The config tree.

    Returns:
        The dataset.
    """
    from lead.policy.transfuser.transfuser import Transfuser

    return Transfuser(lead_config).build_dataset()


def _declared_dtypes(annotation: object) -> tuple[str, ...]:
    """The dtype names a field's jaxtyping annotation allows.

    Args:
        annotation: The dataclass field annotation.

    Returns:
        The allowed dtype names, empty when the field is not an array field.
    """
    for arg in typing.get_args(annotation) or (annotation,):
        dtypes = getattr(arg, "dtypes", None)
        if dtypes:
            return tuple(dtypes)
    return ()


def test_every_populated_field_matches_its_declared_dtype(dataset) -> None:
    """A field the builders fill must carry the dtype the sample declares."""
    fields = {f.name: f.type for f in dataclasses.fields(TransfuserTrainingSample)}
    checked = 0
    for index in range(min(MAX_SAMPLES, len(dataset))):
        sample = dataset[index]
        for name, annotation in fields.items():
            value = getattr(sample, name, None)
            allowed = _declared_dtypes(annotation)
            if value is None or not allowed or not isinstance(value, np.ndarray):
                continue
            assert str(value.dtype) in allowed, (
                f"sample {index} field {name!r}: dtype {value.dtype} "
                f"but the contract declares {allowed}"
            )
            checked += 1
    assert checked, "no array fields were populated; the dataset built nothing"


def test_sample_shapes_follow_the_configured_geometry(dataset, lead_config) -> None:
    """The rasters must match the grid sizes the config derives."""
    transfuser = lead_config.policy.transfuser
    sample = dataset[0]

    if sample.rasterized_lidar is not None:
        assert sample.rasterized_lidar.shape == (
            1,
            transfuser.lidar_height_pixel,
            transfuser.lidar_width_pixel,
        )
    if sample.center_net_heatmap is not None:
        cell_rows = transfuser.lidar_height_pixel // transfuser.bev_downsample_factor
        cell_cols = transfuser.lidar_width_pixel // transfuser.bev_downsample_factor
        assert sample.center_net_heatmap.shape == (
            transfuser.num_box_classes,
            cell_rows,
            cell_cols,
        )
    if sample.rgb is not None:
        assert sample.rgb.shape[0] == 3


def test_collate_gives_every_field_a_leading_batch_axis(dataset) -> None:
    """Collation must not change rank beyond adding the batch axis."""
    samples = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = TransfuserTrainingSample.collate(samples)
    for name, value in vars(samples[0]).items():
        if not isinstance(value, np.ndarray) or name == "extras":
            continue
        assert batch[name].shape == (len(samples), *value.shape), (
            f"{name!r} collated to {tuple(batch[name].shape)}, "
            f"expected {(len(samples), *value.shape)}"
        )


def test_driving_meta_carries_every_required_key(dataset) -> None:
    """A log missing a key the readers index directly cannot be loaded at all."""
    loader = dataset._scene_loader  # noqa: SLF001 — the loader owns the log access
    required = set(RequiredDrivingMeta.__annotations__)
    for index in range(min(MAX_SAMPLES, len(loader.scenes))):
        meta = loader._read_driving_meta(loader.scenes[index], 0)  # noqa: SLF001
        assert meta is not None, f"scene {index} has no driving-meta record"
        missing = required - set(meta)
        assert not missing, (
            f"scene {index} driving-meta is missing required keys: {sorted(missing)}"
        )
