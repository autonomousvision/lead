# ruff: noqa: T201 — a CI check reports to stdout by design.
"""Coverage and equivalence check of the training cache store.

Every sampled normal-view scene must be served from the store (a missing
sample falls back to live computation silently, so training alone cannot
catch it), and stored tensors must decode to what the builders produce live.
Run from the repo root inside the lead environment.
"""

import numpy as np

from lead.api.abstract_policy import build_policy
from lead.api.py123d_log_api import NORMAL_SENSOR_VIEW
from lead.config import load_lead_config

COVERAGE_SAMPLES = 48
EQUIVALENCE_SAMPLES = 4
# Rasterized label maps may move single boundary pixels between builder
# versions; anything above this fraction is a real content change.
BOUNDARY_PIXEL_BUDGET = 0.001


def main() -> None:
    """Check store coverage, then decoded-tensor equivalence on a few samples."""
    config = load_lead_config(
        loaded_config={"training": {"data": {"read_from_cache_store": True}}},
    )
    dataset = build_policy(config).build_dataset()
    count = len(dataset)
    indices = sorted(
        set(np.linspace(0, count - 1, COVERAGE_SAMPLES, dtype=int).tolist()),
    )
    print(f"dataset: {count} samples, checking {len(indices)}")

    uncovered = [
        index
        for index in indices
        if not dataset._read_cached_part_outputs(index, NORMAL_SENSOR_VIEW)
    ]
    if uncovered:
        raise SystemExit(
            f"{len(uncovered)}/{len(indices)} samples are not in the store and "
            f"would silently compute live, starting with indices {uncovered[:5]}",
        )
    print(f"coverage ok: {len(indices)}/{len(indices)} samples served from the store")

    # The lidar raster is stored quantized; everything else must match exactly.
    lidar_atol = 1.0 / config.policy.transfuser.max_lidar_points_per_bev_pixel
    cached_part_names = dataset._cached_part_names
    for index in indices[:EQUIVALENCE_SAMPLES]:
        cached = dataset._read_cached_part_outputs(index, NORMAL_SENSOR_VIEW)
        live = dataset._assemble(cached_part_names, index, use_perturbated_view=False)
        for name, cached_value in cached.items():
            cached_array = np.asarray(cached_value)
            live_array = np.asarray(live[name])
            if name == "rasterized_lidar":
                np.testing.assert_allclose(
                    cached_array,
                    live_array,
                    atol=lidar_atol,
                    err_msg=f"sample {index}: {name}",
                )
            elif name in ("semantic", "bev_semantic"):
                mismatch = np.mean(cached_array != live_array)
                assert mismatch <= BOUNDARY_PIXEL_BUDGET, (
                    f"sample {index}: {name} differs on {mismatch:.2%} of "
                    f"pixels, budget is {BOUNDARY_PIXEL_BUDGET:.2%}"
                )
            else:
                np.testing.assert_array_equal(
                    cached_array,
                    live_array,
                    err_msg=f"sample {index}: {name}",
                )
        print(f"equivalence ok: sample {index}, {len(cached)} tensors")


if __name__ == "__main__":
    main()
