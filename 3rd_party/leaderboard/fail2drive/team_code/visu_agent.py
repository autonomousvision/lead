"""
Simplified visualization agent:
- Drives like AutoPilot
- Displays front RGB live in a window (no disk writes)
"""

import cv2

try:
  import pygame
except ImportError as exc:
  raise RuntimeError("cannot import pygame, make sure pygame package is installed") from exc

from autopilot import AutoPilot


def get_entry_point():
  return "VisuAgent"


class VisuAgent(AutoPilot):
  """AutoPilot variant that shows front RGB in a live window."""

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index, traffic_manager=None)
    self._interface = None
    self._quit_requested = False
    self.visualize = True # Used in autopilot for de

  def sensors(self):
    result = super().sensors()

    result += [{
        "type": "sensor.camera.rgb",
        "x": self.config.camera_pos[0]-4,
        "y": self.config.camera_pos[1],
        "z": self.config.camera_pos[2]+1.5,
        "roll": self.config.camera_rot_0[0],
        "pitch": self.config.camera_rot_0[1]-8,
        "yaw": self.config.camera_rot_0[2],
        "width": 1920,
        "height": 1080,
        "fov": 110,
        "id": "rgb",
    }]

    return result

  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    control = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)

    if not self._quit_requested:
      rgb = input_data["rgb"][1][:, :, :3]
      self._visualize(rgb)

    return control

  def _visualize(self, rgb_img):
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    if self._interface is None:
      self._interface = _VisuInterface(rgb_img.shape[1], rgb_img.shape[0])

    self._interface.run_interface(rgb_img)
    if self._interface.quit_requested:
      self._quit_requested = True

  def destroy(self, results=None):
    if self._interface is not None:
      self._interface.close()
    super().destroy(results)


class _VisuInterface:
  """Minimal pygame interface that displays one RGB image per step."""

  def __init__(self, width, height):
    self._width = width
    self._height = height
    self.quit_requested = False

    pygame.init()
    self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Visu Agent")

  def run_interface(self, image):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self.quit_requested = True
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        self.quit_requested = True

    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    self._display.blit(surface, (0, 0))
    pygame.display.flip()

  def close(self):
    pygame.quit()
