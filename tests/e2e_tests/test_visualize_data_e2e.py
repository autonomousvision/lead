"""The ground-truth visualizer renders every configuration the trainer runs in.

The visualizer indexes the batch the model-specific loader produced, so a label
a configuration switches off must leave it drawing rather than raising.
"""

import numpy as np
import pytest

# One entry per configuration the trainer is run in; each overrides the policy
# switches that change which labels the loader builds.
CONFIGURATIONS = {
    # Perception pretraining: no planning decoder, so no route or waypoints.
    "pretraining": {"use_planning_decoder": False},
    # Posttraining: the planning decoder needs route and waypoint labels.
    "planning_decoder": {"use_planning_decoder": True},
    # The planning decoder without the route smoothing its labels normally get.
    "planning_decoder_unsmoothed_route": {
        "use_planning_decoder": True,
        "smooth_route": False,
    },
    # The BEV semantic overlay is what blends label colours into the canvas, so
    # without it the canvas reaches the drawing primitives untouched.
    "no_bev_semantic": {
        "use_planning_decoder": True,
        "use_bev_semantic": False,
    },
}


def _dataset(overrides: dict, visualize: bool = True):
    """A TransFuser dataset in one configuration.

    Args:
        overrides: ``policy.transfuser`` switches to set.
        visualize: Whether to enable dataset visualization, which builds the
            planning targets in its own right.

    Returns:
        The dataset and the config it was built from.
    """
    from lead.config import load_lead_config
    from lead.policy.transfuser.transfuser import Transfuser

    lead_config = load_lead_config(
        {
            "training": {"experiment": {"visualize_dataset": visualize}},
            "policy": {"transfuser": dict(overrides)},
        },
    )
    return Transfuser(lead_config).build_dataset(), lead_config


@pytest.mark.e2e
@pytest.mark.parametrize("configuration", sorted(CONFIGURATIONS))
def test_visualizer_renders_frame(data_root, configuration: str) -> None:
    """Render a frame from real data in one trainer configuration."""
    from lead.policy.transfuser.visualization.ground_truth_visualizer import (
        GroundTruthVisualizer,
    )

    dataset, lead_config = _dataset(CONFIGURATIONS[configuration])
    # The dataset yields a typed sample; its own collate produces the batch the
    # visualizer indexes.
    batch = dataset.collate_fn([dataset[0]])

    image = GroundTruthVisualizer(lead_config, data=batch).visualize()

    assert image is not None
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8
    assert image.any(), f"{configuration}: the visualizer rendered an empty frame"


@pytest.mark.e2e
@pytest.mark.parametrize("configuration", sorted(CONFIGURATIONS))
def test_planning_labels_follow_the_planning_decoder(
    data_root,
    configuration: str,
) -> None:
    """Route and waypoint labels are built exactly when the decoder needs them."""
    overrides = CONFIGURATIONS[configuration]
    # Visualization builds the planning targets in its own right, so it has to
    # be off for the decoder switch to be the only thing gating them.
    sample = _dataset(overrides, visualize=False)[0][0]

    for field in ("route", "future_waypoints"):
        value = getattr(sample, field, None)
        if overrides["use_planning_decoder"]:
            assert value is not None, (
                f"{configuration}: {field!r} is missing though the planning "
                f"decoder is on"
            )
        else:
            assert value is None, (
                f"{configuration}: {field!r} was built for a run with no "
                f"planning decoder"
            )


@pytest.mark.e2e
def test_visualization_builds_the_planning_targets_on_its_own(data_root) -> None:
    """The visualizer draws route and waypoints, so it needs them without the decoder."""
    sample = _dataset({"use_planning_decoder": False}, visualize=True)[0][0]

    assert sample.route is not None
    assert sample.future_waypoints is not None
