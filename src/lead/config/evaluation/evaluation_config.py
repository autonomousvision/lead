"""Evaluation section of the config tree: driving with the learned agent."""

import os
import pathlib

from lead.config.evaluation.controller_config import ControllerConfig
from lead.config.evaluation.heuristics_config import HeuristicsConfig
from lead.config.evaluation.inference_config import InferenceConfig
from lead.config.node import ConfigNode, child_node, overridable_property


class EvaluationConfig(ConfigNode):
    """Inference driving, weather overrides and which debug/demo outputs are produced."""

    inference = child_node(InferenceConfig)
    controller = child_node(ControllerConfig)
    heuristics = child_node(HeuristicsConfig)

    # --- Evaluation Visualization Settings ---
    # If not None, set a custom weather for evaluation, e.g. "ClearNoon"
    custom_weather: str | None = None
    # If true, use random weather
    random_weather: bool = False

    # Frequency of frame production during evaluation
    produce_frame_frequency: int = 1

    @overridable_property
    def produce_demo_image(self) -> bool:
        """If true produce demo image output."""
        return False

    @overridable_property
    def produce_demo_video(self) -> bool:
        """If true produce demo video output."""
        return False

    @overridable_property
    def produce_debug_image(self) -> bool:
        """If true produce debug image output."""
        return False

    @overridable_property
    def produce_debug_video(self) -> bool:
        """If true produce debug video output."""
        return self._root.debug_mode

    @overridable_property
    def produce_input_image(self) -> bool:
        """If true produce input image output."""
        return False

    @overridable_property
    def produce_input_video(self) -> bool:
        """If true produce input video output."""
        return False

    @overridable_property
    def produce_grid_image(self) -> bool:
        """If true produce grid image output (demo + input stacked vertically)."""
        return False

    @overridable_property
    def produce_grid_video(self) -> bool:
        """If true produce grid video output (demo + input stacked vertically)."""
        return False

    @overridable_property
    def produce_input_log(self) -> bool:
        """If true produce input logging."""
        return False

    @property
    def save_path(self) -> pathlib.Path | None:
        """Get and create the save path for outputs."""
        save_path = os.environ.get("SAVE_PATH")
        if save_path is not None:
            save_path = pathlib.Path(save_path)
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        return save_path

    @property
    def _existing_save_path(self) -> pathlib.Path:
        """The save path, required to exist for the output-path properties."""
        save_path = self.save_path
        assert save_path is not None, "SAVE_PATH must be set"
        return save_path

    @property
    def is_bench2drive(self) -> bool:
        """Check if running in Bench2Drive environment."""
        return os.getenv("IS_BENCH2DRIVE", "0") == "1"

    @property
    def route_id(self) -> str:
        """Get the benchmark route ID from environment."""
        return os.environ["BENCHMARK_ROUTE_ID"]

    @property
    def debug_video_path(self) -> str:
        """Get the path for video output."""
        return os.path.join(self._existing_save_path, f"{self.route_id}_debug.mp4")

    @property
    def demo_video_path(self) -> str:
        """Get the path for demo video output."""
        return os.path.join(self._existing_save_path, f"{self.route_id}_demo.mp4")

    @property
    def input_log_path(self) -> str:
        """Get and create the path for input logging."""
        path = os.path.join(self._existing_save_path, "input_log")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def input_video_path(self) -> str:
        """Get the path for input video output."""
        return os.path.join(self._existing_save_path, f"{self.route_id}_input.mp4")

    @property
    def grid_video_path(self) -> str:
        """Get the path for grid video output."""
        return os.path.join(self._existing_save_path, f"{self.route_id}_grid.mp4")

    @overridable_property
    def video_fps(self) -> float:
        """Calculate video FPS based on frame production frequency."""
        return 20 / self.produce_frame_frequency
