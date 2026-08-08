"""Tests of the cache store: build sample tensors, read them back."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from lead.cache.cache_store import (
    CacheStoreReader,
    CacheStoreWriter,
    check_cache_finger_print,
    encode_tensor,
    read_manifest,
    write_manifest,
)

_CACHED_TENSORS = {
    "rasterized_lidar": "raw",
    "class_raster": {"name": "png", "quantization_scale": 1},
    "town": "raw",
}
_FINGER_PRINT = {"bev_pixels_per_meter": "4.0"}


def _build_store(
    cache_root: Path,
    cache_finger_print: dict[str, str] = _FINGER_PRINT,
) -> dict[str, object]:
    tensors: dict[str, object] = {
        "rasterized_lidar": np.random.rand(1, 8, 8).astype(np.float32),
        "class_raster": np.arange(64, dtype=np.uint8).reshape(8, 8),
        "town": "Town01",
    }
    write_manifest(
        cache_root,
        {
            "tensors": _CACHED_TENSORS,
            "views": ["normal_view"],
            "cache_finger_print": cache_finger_print,
        },
    )
    writer = CacheStoreWriter(cache_root)
    for name, value in tensors.items():
        blob = encode_tensor(value, _CACHED_TENSORS[name])
        writer.write("normal_view", "Town01_log", "1000", name, blob)
    writer.close()
    return tensors


def test_round_trip(tmp_path: Path) -> None:
    tensors = _build_store(tmp_path)
    reader = CacheStoreReader(tmp_path, _FINGER_PRINT)

    assert set(reader.tensor_names) == set(_CACHED_TENSORS)
    assert reader.has_sensor_view("normal_view")
    assert not reader.has_sensor_view("perturbated_view")

    decoded = reader.read("normal_view", "Town01_log", "1000", list(tensors))
    assert decoded is not None
    assert set(decoded) == set(tensors)
    for name, value in tensors.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(decoded[name], value)
            assert decoded[name].dtype == value.dtype
        else:
            assert decoded[name] == value
    reader.close()


def test_half_stored_sample_raises(tmp_path: Path) -> None:
    _build_store(tmp_path)
    reader = CacheStoreReader(tmp_path, _FINGER_PRINT)
    with pytest.raises(KeyError, match="never_built"):
        reader.read(
            "normal_view",
            "Town01_log",
            "1000",
            ["rasterized_lidar", "never_built"],
        )
    reader.close()


def test_sample_the_store_does_not_hold_reads_as_none(tmp_path: Path) -> None:
    tensors = _build_store(tmp_path)
    reader = CacheStoreReader(tmp_path, _FINGER_PRINT)
    assert reader.read("normal_view", "Town01_log", "9999", list(tensors)) is None
    reader.close()


def test_store_without_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_cache"):
        CacheStoreReader(tmp_path, _FINGER_PRINT)


def test_manifest_round_trip(tmp_path: Path) -> None:
    _build_store(tmp_path)
    manifest = read_manifest(tmp_path)
    assert manifest is not None
    assert manifest["views"] == ["normal_view"]
    assert manifest["tensors"]["rasterized_lidar"] == "raw"


def test_reader_survives_pickling_like_a_dataloader_fork(tmp_path: Path) -> None:
    tensors = _build_store(tmp_path)
    reader = CacheStoreReader(tmp_path, _FINGER_PRINT)
    reader.read("normal_view", "Town01_log", "1000", ["town"])

    forked = pickle.loads(pickle.dumps(reader))
    decoded = forked.read("normal_view", "Town01_log", "1000", ["class_raster"])
    assert decoded is not None
    np.testing.assert_array_equal(decoded["class_raster"], tensors["class_raster"])
    reader.close()
    forked.close()


def test_reader_raises_on_finger_print_mismatch(tmp_path: Path) -> None:
    _build_store(tmp_path, cache_finger_print={"bev_pixels_per_meter": "4.0"})
    with pytest.raises(ValueError, match="bev_pixels_per_meter"):
        CacheStoreReader(tmp_path, {"bev_pixels_per_meter": "2.0"})


def test_check_cache_finger_print_ignores_a_store_never_built() -> None:
    check_cache_finger_print(None, {"bev_pixels_per_meter": "2.0"})


def test_check_cache_finger_print_passes_on_a_match() -> None:
    manifest = {"tensors": {}, "views": [], "cache_finger_print": _FINGER_PRINT}
    check_cache_finger_print(manifest, dict(_FINGER_PRINT))


def test_check_cache_finger_print_raises_on_a_changed_value() -> None:
    manifest = {
        "tensors": {},
        "views": [],
        "cache_finger_print": {"bev_pixels_per_meter": "4.0", "num_yaw_bins": "12"},
    }
    with pytest.raises(ValueError, match="bev_pixels_per_meter") as excinfo:
        check_cache_finger_print(
            manifest,
            {"bev_pixels_per_meter": "2.0", "num_yaw_bins": "12"},
        )
    assert "num_yaw_bins" not in str(excinfo.value)


def test_check_cache_finger_print_raises_on_a_new_or_dropped_key() -> None:
    manifest = {"tensors": {}, "views": [], "cache_finger_print": {"a": "1"}}
    with pytest.raises(ValueError, match="b"):
        check_cache_finger_print(manifest, {"a": "1", "b": "2"})
