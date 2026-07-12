"""Debug visualization configuration of the expert (colors and toggles)."""

from lead.config.node import ConfigNode


class ExpertVisualizationConfig(ConfigNode):
    """Debug drawing colors (RGB triplets) and visualization toggles."""

    # Time in seconds to draw the things during debugging
    draw_life_time: float = 0.051

    # --- Debug Colors for Visualization (r, g, b) ---
    # Color for future route visualization during debugging
    future_route_color: list[int] = [0, 255, 0]
    # Color for other vehicles' forecasted bounding boxes
    other_vehicles_forecasted_bbs_color: list[int] = [0, 0, 255]
    # Color for leading vehicle visualization
    leading_vehicle_color: list[int] = [255, 0, 0]
    # Color for trailing vehicle visualization
    trailing_vehicle_color: list[int] = [255, 255, 255]
    # Color for ego vehicle bounding box
    ego_vehicle_bb_color: list[int] = [0, 0, 0]
    # Color for pedestrian forecasted bounding boxes
    pedestrian_forecasted_bbs_color: list[int] = [0, 0, 255]
    # Color for red traffic lights
    red_traffic_light_color: list[int] = [0, 255, 0]
    # Color for yellow traffic lights
    yellow_traffic_light_color: list[int] = [255, 255, 0]
    # Color for green traffic lights
    green_traffic_light_color: list[int] = [255, 0, 0]
    # Color for off traffic lights
    off_traffic_light_color: list[int] = [0, 0, 0]
    # Color for unknown traffic lights
    unknown_traffic_light_color: list[int] = [0, 0, 0]
    # Color for cleared stop signs
    cleared_stop_sign_color: list[int] = [0, 255, 0]
    # Color for uncleared stop signs
    uncleared_stop_sign_color: list[int] = [255, 0, 0]
    # Color for ego vehicle forecasted bounding boxes in hazard situations
    ego_vehicle_forecasted_bbs_hazard_color: list[int] = [255, 0, 0]
    # Color for ego vehicle forecasted bounding boxes in normal situations
    ego_vehicle_forecasted_bbs_normal_color: list[int] = [0, 255, 0]
    # Color for highlighted route segments
    highlight_route_segment_color: list[int] = [255, 0, 0]
    # Color for source lane visualization
    source_lane_color: list[int] = [0, 255, 0]
    # Color for target lane visualization
    target_lane_color: list[int] = [255, 0, 0]
    # Color for opponent traffic route
    opponent_traffic_route_color: list[int] = [255, 0, 0]
    # Color for intersection points
    intersection_point_color: list[int] = [0, 0, 255]
    # Color for adversarial situations
    adversarial_color: list[int] = [255, 0, 0]

    # --- Debug visualization toggles (all off by default; override via env/CLI) ---
    # If true source lane should be visualized.
    visualize_source_lane: bool = False
    # If true target lane should be visualized.
    visualize_target_lane: bool = False
    # If true adversarial elements should be visualized.
    visualize_adversarial: bool = False
    # If true planning area should be visualized.
    visualize_planning_area: bool = False
    # If true radar should be visualized.
    visualize_radar: bool = False
    # If true visualize target points during debugging.
    visualize_target_points: bool = False
    # If true visualize the route.
    visualize_route: bool = False
    # If true visualize the original route.
    visualize_original_route: bool = False
    # If true visualize bounding box IDs during debugging.
    visualize_bb_ids: bool = False
    # If true visualize internal data.
    visualize_internal_data: bool = False
    # If true bounding boxes should be visualized.
    visualize_bounding_boxes: bool = False
    # If true traffic light bounding boxes should be visualized.
    visualize_traffic_lights_bounding_boxes: bool = False
