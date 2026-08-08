"""Named array layouts shared across the loader, the policies and the visualizers.

Aliases, not distinct types: two layouts with the same dtype and shape are
interchangeable to the checkers, so the name documents the frame and the column
meaning that the shape cannot.
"""

import numpy as np
from jaxtyping import Float32, Float64, UInt8

# --- View-frame geometry: everything downstream of ``geometry.to_local_frame_2d``,
# whose float64 rotation matrix sets the width. ---
# One planar point, e.g. a route target point.
Point2D = Float64[np.ndarray, " 2"]
# A planar polyline, e.g. the ego's future waypoints or the route ahead.
Points2D = Float64[np.ndarray, "n_points 2"]
# A pose as a homogeneous transform in the global frame.
SE3Matrix = Float64[np.ndarray, "4 4"]

# --- Rasters ---
# The per-camera images concatenated left to right, in channel-first layout.
StitchedRgb = UInt8[np.ndarray, "3 img_h img_w"]
# BEV class grid, of ``BEVSemanticClass`` values.
BevSemanticGrid = UInt8[np.ndarray, "bev_h bev_w"]
# BEV lidar point density, normalized to [0, 1].
BevDensityGrid = Float32[np.ndarray, "bev_h bev_w"]

# --- Box rows: columns are ``BoundingBoxIndex``. The two frames are not
# interchangeable; convert with ``label_builders.box_rows_to_bev_pixel_frame``. ---
BoxRowsViewFrame = Float32[np.ndarray, "n_boxes 9"]
BoxRowsPixelFrame = Float32[np.ndarray, "n_boxes 9"]
