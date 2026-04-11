"""
Visualization agent:
- Drives like AutoPilot
- Captures a front-facing RGB stream
- Saves frames into the VIZ_PATH directory exported by slurm_evaluate.py
"""

import os
from pathlib import Path

import cv2

from autopilot import AutoPilot


def get_entry_point():
  return "VizPathAgent"


class VizPathAgent(AutoPilot):
  """AutoPilot variant that saves front-facing RGB frames to disk."""

  SAVE_INTERVAL = 10

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index, traffic_manager=None)
    self.visualize = True
    self._viz_path = None

    viz_path = os.environ.get("VIZ_PATH")
    if viz_path:
      self._viz_path = Path(viz_path)
      self._viz_path.mkdir(parents=True, exist_ok=True)

  def sensors(self):
    result = super().sensors()

    result.append({
        "type": "sensor.camera.rgb",
        "x": self.config.camera_pos[0] - 4,
        "y": self.config.camera_pos[1],
        "z": self.config.camera_pos[2] + 1.5,
        "roll": self.config.camera_rot_0[0],
        "pitch": self.config.camera_rot_0[1] - 8,
        "yaw": self.config.camera_rot_0[2],
        "width": 1920,
        "height": 1080,
        "fov": 110,
        "id": "rgb",
    })

    return result

  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    control = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)

    if self._viz_path is not None and "rgb" in input_data and self.step % self.SAVE_INTERVAL == 0:
      rgb = input_data["rgb"][1][:, :, :3]
      cv2.imwrite(str(self._viz_path / f"{self.step:05}.jpg"), rgb)

    return control
