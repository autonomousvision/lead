"""End-to-end test for visualization with real data."""

import pytest


@pytest.mark.e2e
@pytest.mark.skip(reason="Requires real LEAD dataset")
def test_visualizer_renders_frame():
    """Render a frame from real data."""
    from torch.utils.data import default_collate

    from lead.config import load_lead_config
    from lead.policy.transfuser.dataloader.dataset import TransfuserDataset
    from lead.policy.transfuser.visualization.ground_truth_visualizer import (
        GroundTruthVisualizer,
    )

    config = load_lead_config()
    config.training.experiment.visualize_dataset = True

    dataset = TransfuserDataset(config)
    sample = default_collate([dataset[0]])

    visualizer = GroundTruthVisualizer(config, data=sample)
    image = visualizer.visualize()

    assert image is not None
    assert image.shape[2] == 3
