import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from lead.config.policy.transfuser.label_classes import BoundingBoxIndex
from lead.policy.transfuser.decoder.center_net_decoder import PredictedBoundingBox

MARKER_MIN_RADIUS_PIXEL = 3
MARKER_STROKE_RATIO = 0.25


def depth_to_color(
    depth: jt.Float[npt.NDArray, "h w"],
    max_depth_meter: float,
) -> jt.UInt8[npt.NDArray, "h w 3"]:
    """Colorize a metric depth map, near is warm and far is cold.

    Args:
        depth: Metric depth in meters.
        max_depth_meter: Depth mapped to the far end of the colormap. Fixing it
            instead of using the per-frame maximum keeps a color comparable
            across frames.

    Returns:
        The colorized depth image (RGB).
    """
    normalized = np.clip(depth / max_depth_meter, 0.0, 1.0)
    colored = cv2.applyColorMap(
        (255 * (1.0 - normalized)).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _marker_geometry(size: int) -> tuple[int, int]:
    """Return the (radius, stroke thickness) in pixels for a marker of the given size."""
    radius = max(MARKER_MIN_RADIUS_PIXEL, size)
    return radius, max(1, round(radius * MARKER_STROKE_RATIO))


def draw_ring(
    bev: jt.UInt8[npt.NDArray, "h w 3"],
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a ring of the given outer radius in pixels."""
    radius, thickness = _marker_geometry(size)
    cv2.circle(bev, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)


def draw_cross(
    bev: jt.UInt8[npt.NDArray, "h w 3"],
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a diagonal cross of the given half-diagonal in pixels."""
    radius, thickness = _marker_geometry(size)
    arm = int(round(radius / np.sqrt(2)))
    cv2.line(bev, (x - arm, y - arm), (x + arm, y + arm), color, thickness, cv2.LINE_AA)
    cv2.line(bev, (x - arm, y + arm), (x + arm, y - arm), color, thickness, cv2.LINE_AA)


def draw_box(
    img: jt.UInt8[npt.NDArray, "H W C"],
    box: jt.Float[npt.NDArray, " N"] | PredictedBoundingBox,
    color: list[float] | tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    corner_radius: int = 2,
) -> jt.UInt8[npt.NDArray, "H W C"]:
    """Utility function to draw a rotated bounding box on BEV image with rounded corners.

    Args:
        img: The BEV image to draw on.
        box: The bounding box parameters [center_x, center_y, width, height, yaw, velocity] or PredictedBoundingBox.
        color: Color of the box (BGR).
        thickness: Thickness of the box lines.
        corner_radius: Radius of the rounded corners.
    Returns:
        img: The BEV image with the box drawn.
    """
    translation = np.array([[box[1], box[0]]])
    width = box[BoundingBoxIndex.W]
    height = box[BoundingBoxIndex.H]
    yaw = -box[BoundingBoxIndex.YAW] + np.pi / 2
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    corners = np.array(
        [[-width, -height], [width, -height], [width, height], [-width, height]],
    )
    corner_global = (rot @ corners.T).T + translation
    corner_global = corner_global.astype(int)

    # Draw edges with rounded corners
    for i in range(4):
        r0, c0 = corner_global[i]
        r1, c1 = corner_global[(i + 1) % 4]
        r2, c2 = corner_global[(i + 2) % 4]

        # Calculate shortened line segment (leave space for arc)
        vec1 = np.array([c1 - c0, r1 - r0])
        vec2 = np.array([c2 - c1, r2 - r1])

        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)

        if len1 > corner_radius and len2 > corner_radius:
            # Shorten lines by corner_radius
            unit1 = vec1 / len1
            unit2 = vec2 / len2

            start_point = (
                int(c0 + unit1[0] * corner_radius),
                int(r0 + unit1[1] * corner_radius),
            )
            end_point = (
                int(c1 - unit1[0] * corner_radius),
                int(r1 - unit1[1] * corner_radius),
            )

            # Draw the shortened line
            cv2.line(
                img,
                start_point,
                end_point,
                color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            # Draw arc at corner
            # Calculate arc parameters
            center = (c1, r1)
            start_angle = np.degrees(np.arctan2(-unit1[1], -unit1[0]))
            end_angle = np.degrees(np.arctan2(unit2[1], unit2[0]))

            # Handle angle wraparound
            if end_angle - start_angle > 180:
                end_angle -= 360
            elif start_angle - end_angle > 180:
                start_angle -= 360

            cv2.ellipse(
                img,
                center,
                (corner_radius, corner_radius),
                0,
                start_angle,
                end_angle,
                color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    # Draw speed line from center, but only if the box carries a velocity
    # component. Future-pose footprint boxes built from waypoints are just
    # [x, y, w, h, yaw] (no velocity), so skip the speed line for them.
    has_velocity = not isinstance(box, np.ndarray) or len(box) > int(
        BoundingBoxIndex.VELOCITY,
    )
    if has_velocity:
        # Calculate center of the box (translation is [row, col])
        center_x = int(translation[0][1])  # col (x in image)
        center_y = int(translation[0][0])  # row (y in image)

        # Scale factor: pixels per m/s (adjust as needed)
        scale_factor = 5.0
        line_length = box[BoundingBoxIndex.VELOCITY] * scale_factor

        # Calculate direction vector using the same rotation as the box
        # The direction vector starts at [line_length, 0] and is rotated by yaw
        direction = rot @ np.array([[line_length], [0]])

        # Calculate end point (direction is [row_offset, col_offset])
        end_x = int(center_x + direction[1][0])  # col + col_offset
        end_y = int(center_y + direction[0][0])  # row + row_offset

        # Draw speed line (OpenCV uses (x, y) = (col, row) format)
        cv2.line(
            img,
            (center_x, center_y),
            (end_x, end_y),
            (255, 0, 0),  # Red color for speed
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    return img


def draw_circle_with_number(
    bev_image: jt.UInt8[npt.NDArray, "h w 3"],
    x: int,
    y: int,
    color: tuple[int, int, int],
    radius: int,
    number: int,
):
    """Draw a circle with a number inside.

    Args:
        bev_image: The BEV image to draw on.
        x: X coordinate of the circle center.
        y: Y coordinate of the circle center.
        color: Color of the circle (BGR).
        radius: Radius of the circle.
        number: Number to draw inside the circle.
    """
    # Draw filled circle
    cv2.circle(bev_image, (x, y), radius, color, thickness=-1)

    # Draw white number in the center (rotated 90 degrees)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text = str(number)

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Create a temporary image for the text
    temp_img = np.zeros((text_size[1] + 10, text_size[0] + 10, 3), dtype=np.uint8)
    cv2.putText(
        temp_img,
        text,
        (5, text_size[1] + 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    # Rotate 90 degrees clockwise
    rotated_text = cv2.rotate(temp_img, cv2.ROTATE_90_CLOCKWISE)

    # Calculate position to center the rotated text
    text_h, text_w = rotated_text.shape[:2]
    text_x = max(0, x - text_w // 2)
    text_y = max(0, y - text_h // 2)

    # Ensure we don't go out of bounds
    text_x_end = min(bev_image.shape[1], text_x + text_w)
    text_y_end = min(bev_image.shape[0], text_y + text_h)

    # Only overlay if there's a valid region to draw on
    if text_x_end > text_x and text_y_end > text_y:
        # Overlay the rotated text
        mask = (
            rotated_text[: text_y_end - text_y, : text_x_end - text_x].sum(axis=2) > 0
        )
        bev_image[text_y:text_y_end, text_x:text_x_end][mask] = rotated_text[
            : text_y_end - text_y,
            : text_x_end - text_x,
        ][mask]


def lighter_shade(
    color: tuple[int, int, int],
    i: int,
    max_len: int,
    max_lighter: int = 100,
) -> tuple[int, int, int]:
    """Create a lighter shade of a color based on position in sequence.

    Args:
        color: RGB color tuple.
        i: Current position in sequence.
        max_len: Maximum length of sequence.
        max_lighter: Maximum lightening factor.

    Returns:
        Lighter RGB color tuple.
    """
    factor = i / max(1, max_len - 1)

    lighter_color = np.clip(
        np.array(color, dtype=np.int32) + factor * max_lighter,
        0,
        255,
    ).astype(int)
    return (int(lighter_color[0]), int(lighter_color[1]), int(lighter_color[2]))
