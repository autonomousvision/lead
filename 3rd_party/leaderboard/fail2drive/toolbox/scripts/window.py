"""
Route Creation and Scenario Editing Tool for CARLA Simulator

This Python script provides a graphical user interface (GUI) for creating and editing routes,
as well as defining scenarios, within the CARLA driving simulator environment. It allows users
to load, save, and manipulate routes by adding or removing waypoints, defining scenario trigger
points, and specifying scenario attributes. The tool also supports visualizing the map, routes,
and associated elements such as stop signs and traffic lights.
"""

import sys
from PyQt6.QtWidgets import (
    QLabel,
    QMessageBox,
    QFrame,
    QFileDialog,
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QCheckBox,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
)
from PyQt6.QtCore import Qt, QTimer, QPointF, QEventLoop
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QCursor, QGuiApplication
import time
from route_manager import RouteManager
from carla_simulator_client import CarlaClient
import argparse
from map_selection_dialog import MapSelectionDialog
from loading_indicator_window import LoadingIndicatorWindow
import numpy as np
from scipy.spatial import cKDTree
import carla
from scenario_selection_dialog import ScenarioSelectionDialog
import config
from scenario_attribute_dialog import ScenarioAttributeDialog
from custom_obstacle_dialog import CustomObstacleDialog
from permuted_construction_layout_dialog import PermutedConstructionLayoutDialog
from bad_parking_layout_dialog import BadParkingLayoutDialog
from road_blocked_layout_dialog import RoadBlockedLayoutDialog
from scenario_editor_registry import get_scenario_editor_profile
from pathlib import Path


class Separator(QFrame):
    """
    A simple horizontal separator widget.
    """

    def __init__(self, separator_height=2):
        super().__init__()
        self.setFixedHeight(separator_height)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


class Canvas(QWidget):
    """
    A canvas widget for displaying and interacting with the CARLA map, routes, and scenarios.
    """

    def __init__(
        self,
        carla_client,
        fps=20.0,
        max_scaling=200.0,
        default_offset=10.0,
        max_drawn_points=20_000,
        interpolating_after_ticks_of_no_mouse_movement=0.5,
        parent=None,
        route_manager=None,
    ):
        super().__init__(parent)

        self.fps = fps
        self.fps_inv = 1.0 / fps
        self.last_screen_update = time.time()
        self.parent_obj = parent
        self.route_manager = route_manager
        self.carla_client = carla_client
        self.max_scaling = max_scaling
        self.max_drawn_points = max_drawn_points
        self.interpolating_after_ticks_of_no_mouse_movement = interpolating_after_ticks_of_no_mouse_movement

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.scaling = 1.0
        self.global_scaling = 1.0
        self.offset = np.array([0.0, 0.0])
        self.default_offset = default_offset
        self.map_offset = np.array([0.0, 0.0])
        self.panning = False
        self.dragging_waypoint_idx = None
        self.dragging_scenario_idx = None
        self.pending_scenario_click_idx = None
        self.pending_scenario_press_screen_pos = None
        self.ctrl_pressed = False
        self.current_mouse_pos = QPointF(0, 0)
        self.since_last_mouse_movement = time.time()
        self.location_transform_attributes = []
        self.location_transform_on_finish = None
        self.location_transform_anchor = None

        # Load the icons for stop signs and traffic lights
        self.stop_sign_pixmap = QPixmap("scripts/images/stop_sign_icon.png")
        self.traffic_light_pixmap = QPixmap("scripts/images/traffic_light_icon.png")

        self.road_waypoints_np = None
        self.parking_waypoints_np = None
        self.biking_waypoints_np = None
        self.biking_tree = None
        self.traffic_light_centers_np = None
        self.stop_sign_centers_np = None
        self.min_coords = None
        self.map_width = None
        self.map_height = None
        self.map_size = None
        self.closest_map_coord_screen_coords = None
        self.hovered_waypoint_idx = None
        self.hovered_scenario_idx = None
        self.selected_route = None
        self.last_mouse_pos = QPointF(0, 0)

        # Define colors
        self.STOP_SIGN_COLOR = QColor(180, 50, 50)
        self.TRAFFIC_LIGHT_COLOR = QColor(50, 180, 50)
        self.BACKGROUND_COLOR = QColor(255, 255, 255)
        self.MAP_COLOR = QColor(0, 0, 0)
        self.PARKING_LOT_COLOR = QColor(100, 100, 255)
        self.BIKING_LANE_COLOR = QColor(220, 50, 50)
        self.NEW_ROUTE_PART_COLOR = QColor(130, 250, 130)
        self.ROUTE_COLOR = QColor(0, 170, 0)
        self.OTHER_ROUTES_COLOR = QColor(170, 220, 170)
        self.FIRST_WP = QColor(0, 255, 0)
        self.CURSOR_COLOR = QColor(0, 255, 0)
        self.CURSOR_SCENARIO_MODE_COLOR = QColor(220, 60, 60)
        self.WAYPOINT_MOVE_INDICATOR_COLOR = QColor(255, 170, 0)
        self.WAYPOINT_REMOVE_INDICATOR_COLOR = QColor(220, 50, 50)
        self.SCENARIO_COLOR = QColor(255, 100, 100)
        self.SELECTED_SCENARIO_COLOR = QColor(255, 180, 40)
        self.SCENARIO_ATTR_POINT_COLOR = QColor(60, 120, 255)
        self.SCENARIO_ATTR_POINT_SUBTLE_COLOR = QColor(60, 120, 255, 90)

        # Size of drawn circles
        self.ROAD_WPS_SIZE = 0.5
        self.CURSOR_WP_SIZE = 8.0
        self.SCALING_STOP_SIGN = 6.0
        self.SCALING_TRAFFIC_LIGHT = 6.0
        self.SCALING_SPARSE_ROUTE = 8.0
        self.SCALING_DENSE_ROUTE = 4.0
        self.SCALING_OTHER_DENSE_ROUTE = 2.0
        self.SCALING_NEW_ROUTE_PART = 4.0
        self.SCALING_SCENARIO_WAYPOINTS = 8.0
        self.SCALING_SCENARIO_ATTR_WAYPOINTS = 6.0
        self.SCALING_SCENARIO_TEXTS = 16.0

        self.timer = QTimer(self)  # Create a QTimer object
        self.timer.timeout.connect(
            self.update_when_no_movement
        )  # Connect timeout signal to update_when_no_movement method
        self.timer.start(int(self.fps_inv * 1000))  # Start the timer with a timeout based on the FPS
        self.needs_repaint = True

    def request_repaint(self):
        self.needs_repaint = True

    def refresh_hover_and_cursor(self):
        mouse_pos = self.current_mouse_pos.toPoint()
        if self.selected_route is not None:
            mouse_pos_screen_coords = np.array([mouse_pos.x(), mouse_pos.y()], dtype="float")
            mouse_pos_map_coords = self.screen_coords_to_world_coords(mouse_pos_screen_coords[None])[0]
            self.hovered_waypoint_idx = self.selected_route.get_waypoint_index_near(mouse_pos_map_coords)
            self.hovered_scenario_idx = self.selected_route.get_scenario_index_near(mouse_pos_map_coords)
        else:
            self.hovered_waypoint_idx = None
            self.hovered_scenario_idx = None

        if self.hovered_waypoint_idx is not None or self.hovered_scenario_idx is not None:
            self.interpolated_trace = []

        if self.dragging_waypoint_idx is not None or self.dragging_scenario_idx is not None:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif (
            self.ctrl_pressed
            and self.hovered_waypoint_idx is not None
            and not self.parent_obj.add_scenario_mode
            and not self.location_transform_attributes
        ):
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif (
            self.hovered_waypoint_idx is not None
            and not self.parent_obj.add_scenario_mode
            and not self.location_transform_attributes
        ):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif (
            self.hovered_scenario_idx is not None
            and not self.parent_obj.add_scenario_mode
            and not self.location_transform_attributes
        ):
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    def update_when_no_movement(self):
        """
        Updates the canvas when there is no mouse movement for a certain period of time.
        This method interpolates the route between the last waypoint and the closest map coordinate.
        """
        prev_ctrl = self.ctrl_pressed
        prev_hovered_wp = self.hovered_waypoint_idx
        prev_hovered_scenario = self.hovered_scenario_idx

        # Best-practice state sync: derive pointer+modifier state from global inputs every tick.
        self.ctrl_pressed = bool(QGuiApplication.queryKeyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
        self.current_mouse_pos = QPointF(self.mapFromGlobal(QCursor.pos()))
        self.refresh_hover_and_cursor()

        if (
            self.ctrl_pressed != prev_ctrl
            or self.hovered_waypoint_idx != prev_hovered_wp
            or self.hovered_scenario_idx != prev_hovered_scenario
        ):
            self.request_repaint()

        if (
            self.selected_route is not None
            and
            not self.location_transform_attributes
            and not self.parent_obj.add_scenario_mode
            and not self.ctrl_pressed
            and self.dragging_waypoint_idx is None
            and self.dragging_scenario_idx is None
            and self.pending_scenario_click_idx is None
            and self.hovered_waypoint_idx is None
            and self.hovered_scenario_idx is None
            and not self.interpolated_trace
            and self.closest_map_coord_screen_coords is not None
            and time.time() - self.since_last_mouse_movement > self.interpolating_after_ticks_of_no_mouse_movement
        ):
            closest_map_loc = self.screen_coords_to_world_coords(self.closest_map_coord_screen_coords[None])[0]
            closest_map_loc = carla.Location(closest_map_loc[0], closest_map_loc[1])
            self.interpolated_trace = self.selected_route.interpolate_from_last_wp(closest_map_loc)
            self.request_repaint()

        if self.needs_repaint:
            self.update()
            self.needs_repaint = False

    def update_data_from_carla_client(self):
        """
        Updates the map data from the CARLA client, including waypoints, traffic light centers, stop sign centers, and map dimensions.
        """
        self.road_waypoints_np = self.carla_client.road_waypoints_np
        self.parking_waypoints_np = self.carla_client.parking_waypoints_np
        self.biking_waypoints_np = self.carla_client.biking_waypoints_np
        self.traffic_light_centers_np = self.carla_client.traffic_light_centers_np
        self.stop_sign_centers_np = self.carla_client.stop_sign_centers_np
        self.min_coords = self.carla_client.min_coords
        self.map_width = self.carla_client.map_width
        self.map_height = self.carla_client.map_height
        self.map_size = self.carla_client.map_size
        self.interpolated_trace = []

        self.all_waypoints_np = np.concatenate([self.road_waypoints_np, self.parking_waypoints_np], axis=0)
        self.tree = cKDTree(self.all_waypoints_np)
        if self.biking_waypoints_np is not None and self.biking_waypoints_np.shape[0] > 0:
            self.biking_tree = cKDTree(self.biking_waypoints_np)
        else:
            self.biking_tree = None

        self.closest_map_coord_screen_coords = self.road_waypoints_np[0, :2]
        self.request_repaint()

    def update_selected_route(self, selected_route):
        """
        Updates the currently selected route and resets the canvas view.
        """
        self.selected_route = selected_route
        self.request_repaint()

    def reset_map_offset_and_scaling(self):
        if self.map_size is not None:
            window_size = self.size()
            window_size = np.array([window_size.width(), window_size.height()], dtype="float")
            self.map_offset = np.maximum(
                0, (window_size - 2 * self.default_offset - self.scaling * self.global_scaling * self.map_size) / 2
            )
            self.offset = np.array([0.0, 0.0])
            self.scaling = 1.0
            self.update_global_scaling(self.size())
            self.request_repaint()

    def paintEvent(self, event):
        """
        Handles the paintEvent to draw the map, routes, and associated elements on the canvas.
        """
        if not (
            self.selected_route is None
            or self.road_waypoints_np is None
            or self.parking_waypoints_np is None
            or self.traffic_light_centers_np is None
            or self.stop_sign_centers_np is None
        ):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

            road_waypoints = self.world_coords_to_screen_coords(self.road_waypoints_np[:, :2])
            road_waypoints = self.select_coords_inside_window(road_waypoints)
            parking_waypoints = self.world_coords_to_screen_coords(self.parking_waypoints_np[:, :2])
            parking_waypoints = self.select_coords_inside_window(parking_waypoints)
            biking_waypoints = np.zeros((0, 2), dtype="float")
            if (
                self._active_location_attr_type() is not None
                and "bicycle" in self._active_location_attr_type()
                and self.biking_waypoints_np is not None
                and self.biking_waypoints_np.shape[0] > 0
            ):
                biking_waypoints = self.world_coords_to_screen_coords(self.biking_waypoints_np[:, :2])
                biking_waypoints = self.select_coords_inside_window(biking_waypoints)
            sparse_waypoints = np.array(self.selected_route.waypoints).reshape((-1, 3))[:, :2]
            sparse_waypoints = self.world_coords_to_screen_coords(sparse_waypoints)
            sparse_waypoints = self.select_coords_inside_window(sparse_waypoints)
            sparse_waypoints_raw = np.array(self.selected_route.waypoints).reshape((-1, 3))[:, :2]
            sparse_waypoints_raw_screen = self.world_coords_to_screen_coords(sparse_waypoints_raw)
            scenario_trigger_points = np.array(self.selected_route.scenario_trigger_points).reshape((-1, 3))[:, :2]
            scenario_trigger_points = self.world_coords_to_screen_coords(scenario_trigger_points)
            dense_waypoints = np.array(self.selected_route.dense_waypoints).reshape((-1, 3))[:, :2]
            dense_waypoints = self.world_coords_to_screen_coords(dense_waypoints)
            dense_waypoints = self.select_coords_inside_window(dense_waypoints)
            interpolated_trace = np.array(self.interpolated_trace).reshape((-1, 3))[:, :2]
            interpolated_trace = self.world_coords_to_screen_coords(interpolated_trace)
            interpolated_trace = self.select_coords_inside_window(interpolated_trace)

            other_routes_dense_points = np.zeros((0, 2), dtype="float")
            if self.parent_obj.show_other_routes_checkbox.isChecked():
                other_routes = [route for route in self.route_manager.routes.values()]
                other_routes_dense_points = [np.array(route.dense_waypoints).reshape((-1, 3)) for route in other_routes]
                if not other_routes_dense_points:
                    other_routes_dense_points = [[]]
                other_routes_dense_points = np.concatenate(other_routes_dense_points, axis=0).reshape((-1, 3))
                other_routes_dense_points = other_routes_dense_points[:, :2]
                other_routes_dense_points = self.world_coords_to_screen_coords(other_routes_dense_points)
                other_routes_dense_points = self.select_coords_inside_window(other_routes_dense_points)

            n_skip = max(
                1,
                (
                    road_waypoints.shape[0]
                    + parking_waypoints.shape[0]
                    + biking_waypoints.shape[0]
                    + dense_waypoints.shape[0]
                    + interpolated_trace.shape[0]
                    + other_routes_dense_points.shape[0]
                )
                // self.max_drawn_points,
            )
            road_waypoints = road_waypoints[::n_skip]
            parking_waypoints = parking_waypoints[::n_skip]
            biking_waypoints = biking_waypoints[::n_skip]
            dense_waypoints = dense_waypoints[::n_skip]
            interpolated_trace = interpolated_trace[::n_skip]
            other_routes_dense_points = other_routes_dense_points[::n_skip]

            road_waypoints = [QPointF(x, y) for (x, y) in road_waypoints.tolist()]
            parking_waypoints = [QPointF(x, y) for (x, y) in parking_waypoints.tolist()]
            biking_waypoints = [QPointF(x, y) for (x, y) in biking_waypoints.tolist()]
            sparse_waypoints = [QPointF(x, y) for (x, y) in sparse_waypoints.tolist()]
            dense_waypoints = [QPointF(x, y) for (x, y) in dense_waypoints.tolist()]
            scenario_trigger_points_ = [QPointF(x, y) for (x, y) in scenario_trigger_points.tolist()]
            interpolated_trace = [QPointF(x, y) for (x, y) in interpolated_trace.tolist()]
            other_routes_dense_points = [QPointF(x, y) for (x, y) in other_routes_dense_points.tolist()]

            painter.setPen(
                QPen(
                    self.MAP_COLOR,
                    max(1, self.global_scaling * self.scaling * self.ROAD_WPS_SIZE),
                    Qt.PenStyle.DashDotLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawPoints(road_waypoints)

            painter.setPen(
                QPen(
                    self.PARKING_LOT_COLOR,
                    max(1, self.global_scaling * self.scaling * self.ROAD_WPS_SIZE),
                    Qt.PenStyle.DashDotLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawPoints(parking_waypoints)

            if biking_waypoints:
                painter.setPen(
                    QPen(
                        self.BIKING_LANE_COLOR,
                        max(1.5, self.global_scaling * self.scaling * (self.ROAD_WPS_SIZE + 0.7)),
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawPoints(biking_waypoints)

            painter.setPen(
                QPen(
                    self.OTHER_ROUTES_COLOR,
                    max(
                        self.SCALING_OTHER_DENSE_ROUTE,
                        self.global_scaling * self.scaling * self.SCALING_OTHER_DENSE_ROUTE,
                    ),
                    Qt.PenStyle.DashDotLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawPoints(other_routes_dense_points)

            painter.setPen(
                QPen(
                    self.ROUTE_COLOR,
                    max(self.SCALING_DENSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_DENSE_ROUTE),
                    Qt.PenStyle.DashDotLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawPoints(dense_waypoints)

            painter.setPen(
                QPen(
                    self.NEW_ROUTE_PART_COLOR,
                    max(self.SCALING_NEW_ROUTE_PART, self.global_scaling * self.scaling * self.SCALING_NEW_ROUTE_PART),
                    Qt.PenStyle.DashDotLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawPoints(interpolated_trace)

            painter.setPen(
                QPen(
                    self.ROUTE_COLOR,
                    max(self.SCALING_SPARSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_SPARSE_ROUTE),
                    Qt.PenStyle.DashDotLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawPoints(sparse_waypoints)

            if sparse_waypoints:
                painter.setPen(
                    QPen(
                        self.FIRST_WP,
                        max(self.SCALING_SPARSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_SPARSE_ROUTE),
                        Qt.PenStyle.DashDotLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawPoints([sparse_waypoints[0]])

            if self.hovered_waypoint_idx is not None and self.hovered_waypoint_idx < sparse_waypoints_raw_screen.shape[0]:
                move_wp = sparse_waypoints_raw_screen[self.hovered_waypoint_idx]
                move_marker = QPointF(float(move_wp[0]), float(move_wp[1]))
                marker_color = self.WAYPOINT_REMOVE_INDICATOR_COLOR if self.ctrl_pressed else self.WAYPOINT_MOVE_INDICATOR_COLOR
                painter.setPen(
                    QPen(
                        marker_color,
                        max(2.0, self.global_scaling * self.scaling * (self.SCALING_SPARSE_ROUTE + 2.0)),
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawPoint(move_marker)

            # Draw scenario attribute points (location/transform-style params).
            # Render before scenario dots so the main scenario marker stays on top.
            attr_world_xy = []
            attr_meta = []
            scenario_count = min(
                len(self.selected_route.scenarios),
                len(self.selected_route.scenario_trigger_points),
            )
            for scenario_idx in range(scenario_count):
                scenario_elem = self.selected_route.scenarios[scenario_idx]
                trigger_loc = self.selected_route.scenario_trigger_points[scenario_idx]
                for child in scenario_elem:
                    if child.tag == "trigger_point":
                        continue
                    if child.get("x") is None or child.get("y") is None:
                        continue
                    try:
                        attr_x = float(child.get("x"))
                        attr_y = float(child.get("y"))
                    except (TypeError, ValueError):
                        continue
                    attr_world_xy.append([attr_x, attr_y])
                    attr_meta.append((scenario_idx, child.tag, trigger_loc[0], trigger_loc[1]))

            if attr_world_xy:
                attr_screen_xy = self.world_coords_to_screen_coords(np.array(attr_world_xy, dtype="float"))
                p = max(self.SCALING_SPARSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_SPARSE_ROUTE)
                for (ax, ay), (scenario_idx, tag_name, tx, ty) in zip(attr_screen_xy, attr_meta):
                    trigger_screen = self.world_coords_to_screen_coords(np.array([[tx, ty]], dtype="float"))[0]
                    is_hovered_attr = self.hovered_scenario_idx is not None and self.hovered_scenario_idx == scenario_idx

                    line_color = self.SCENARIO_ATTR_POINT_COLOR if is_hovered_attr else self.SCENARIO_ATTR_POINT_SUBTLE_COLOR
                    point_color = self.SCENARIO_ATTR_POINT_COLOR if is_hovered_attr else self.SCENARIO_ATTR_POINT_SUBTLE_COLOR
                    line_width = max(1.0, self.global_scaling * self.scaling * (2.0 if is_hovered_attr else 1.0))
                    point_size = max(
                        (self.SCALING_SCENARIO_ATTR_WAYPOINTS if is_hovered_attr else self.SCALING_SCENARIO_ATTR_WAYPOINTS * 0.55),
                        self.global_scaling
                        * self.scaling
                        * (self.SCALING_SCENARIO_ATTR_WAYPOINTS if is_hovered_attr else self.SCALING_SCENARIO_ATTR_WAYPOINTS * 0.55),
                    )

                    painter.setPen(
                        QPen(
                            line_color,
                            line_width,
                            Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap,
                        )
                    )
                    painter.drawLine(
                        QPointF(float(trigger_screen[0]), float(trigger_screen[1])),
                        QPointF(float(ax), float(ay)),
                    )
                    painter.setPen(
                        QPen(
                            point_color,
                            point_size,
                            Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap,
                        )
                    )
                    painter.drawPoint(QPointF(float(ax), float(ay)))

                    if is_hovered_attr:
                        painter.drawText(int(round(ax + p)), int(round(ay - p)), tag_name)

            painter.setPen(
                QPen(
                    self.SCENARIO_COLOR,
                    max(
                        self.SCALING_SCENARIO_WAYPOINTS,
                        self.global_scaling * self.scaling * self.SCALING_SCENARIO_WAYPOINTS,
                    ),
                    Qt.PenStyle.DashDotLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawPoints(scenario_trigger_points_)

            if self.hovered_scenario_idx is not None and self.hovered_scenario_idx < scenario_trigger_points.shape[0]:
                hx, hy = scenario_trigger_points[self.hovered_scenario_idx]
                scenario_color = self.WAYPOINT_REMOVE_INDICATOR_COLOR if self.ctrl_pressed else self.SCENARIO_COLOR
                painter.setPen(
                    QPen(
                        scenario_color,
                        max(
                            self.SCALING_SCENARIO_WAYPOINTS + 3.0,
                            self.global_scaling * self.scaling * (self.SCALING_SCENARIO_WAYPOINTS + 3.0),
                        ),
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawPoint(QPointF(float(hx), float(hy)))

            selected_ref = self.parent_obj.selected_scenario_ref
            if (
                selected_ref is not None
                and self.parent_obj.route_manager.selected_route_id == selected_ref[0]
                and selected_ref[1] < scenario_trigger_points.shape[0]
            ):
                sx, sy = scenario_trigger_points[selected_ref[1]]
                painter.setPen(
                    QPen(
                        self.SELECTED_SCENARIO_COLOR,
                        max(
                            self.SCALING_SCENARIO_WAYPOINTS + 4.0,
                            self.global_scaling * self.scaling * (self.SCALING_SCENARIO_WAYPOINTS + 4.0),
                        ),
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawPoint(QPointF(float(sx), float(sy)))

            if self.location_transform_attributes and self.location_transform_anchor is not None:
                anchor_vals = np.array(self.location_transform_anchor, dtype="float").flatten()
                if anchor_vals.size >= 2:
                    anchor_xy = anchor_vals[:2][None]
                    anchor_screen = self.world_coords_to_screen_coords(anchor_xy)
                    ax, ay = anchor_screen[0]
                    painter.setPen(
                        QPen(
                            self.SELECTED_SCENARIO_COLOR,
                            max(
                                self.SCALING_SCENARIO_WAYPOINTS + 7.0,
                                self.global_scaling * self.scaling * (self.SCALING_SCENARIO_WAYPOINTS + 7.0),
                            ),
                            Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap,
                        )
                    )
                    painter.drawPoint(QPointF(float(ax), float(ay)))

            factor = max(1, int(self.SCALING_STOP_SIGN * self.scaling * self.global_scaling))
            resized_stop_sign_pixmap = self.stop_sign_pixmap.scaledToHeight(factor)
            stop_sign_locations = self.world_coords_to_screen_coords(self.stop_sign_centers_np)
            stop_sign_locations = self.select_coords_inside_window(stop_sign_locations)
            resized_stop_sign_pixmap_size = np.array(
                [resized_stop_sign_pixmap.size().width(), resized_stop_sign_pixmap.size().height()], dtype="float"
            )
            stop_sign_locations = stop_sign_locations - resized_stop_sign_pixmap_size[None] / 2.0
            for x, y in stop_sign_locations:
                painter.drawPixmap(int(round(x)), int(round(y)), resized_stop_sign_pixmap)

            factor = max(1, int(self.SCALING_TRAFFIC_LIGHT * self.scaling * self.global_scaling))
            resized_traffic_light_pixmap = self.traffic_light_pixmap.scaledToHeight(factor)
            traffic_light_locations = self.world_coords_to_screen_coords(self.traffic_light_centers_np)
            traffic_light_locations = self.select_coords_inside_window(traffic_light_locations)
            resized_traffic_light_pixmap_size = np.array(
                [resized_traffic_light_pixmap.size().width(), resized_traffic_light_pixmap.size().height()],
                dtype="float",
            )
            traffic_light_locations = traffic_light_locations - resized_traffic_light_pixmap_size[None] / 2.0
            for x, y in traffic_light_locations:
                painter.drawPixmap(int(round(x)), int(round(y)), resized_traffic_light_pixmap)

            painter.setPen(
                QPen(
                    self.SCENARIO_COLOR,
                    max(self.SCALING_SCENARIO_TEXTS, self.global_scaling * self.scaling * self.SCALING_SCENARIO_TEXTS),
                    Qt.PenStyle.DashDotLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            scenario_types = self.selected_route.scenario_types
            p = max(self.SCALING_SPARSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_SPARSE_ROUTE)
            for (x, y), scenario_type in zip(scenario_trigger_points, scenario_types):
                painter.drawText(int(round(x + p)), int(round(y - p)), scenario_type)

            if self.hovered_waypoint_idx is None and self.hovered_scenario_idx is None:
                closest_map_coord_screen_coords = [
                    QPointF(self.closest_map_coord_screen_coords[0], self.closest_map_coord_screen_coords[1])
                ]
                cursor_color = self.CURSOR_SCENARIO_MODE_COLOR if self.parent_obj.add_scenario_mode else self.CURSOR_COLOR
                painter.setPen(
                    QPen(
                        cursor_color,
                        self.global_scaling * self.scaling * self.CURSOR_WP_SIZE,
                        Qt.PenStyle.DashDotLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                )
                painter.drawPoints(closest_map_coord_screen_coords)

    def wheelEvent(self, event):
        """
        Handles mouse wheel events for zooming in and out on the canvas.
        """
        self.since_last_mouse_movement = time.time()
        self.interpolated_trace = []

        window_size = self.size()
        window_size = np.array([window_size.width(), window_size.height()], dtype="float")

        self.map_offset = np.maximum(
            0, (window_size - 2 * self.default_offset - self.scaling * self.global_scaling * self.map_size) / 2
        )

        # Increase or decrease the map scale by 10% per mouse wheel spinning event
        scaling = np.clip(self.scaling * (1.0 + 0.1 * event.angleDelta().y() / 120.0), 1.0, self.max_scaling)

        # Zoom in at the location of the cursor
        self.offset += (scaling / self.scaling - 1.0) * (
            self.default_offset + self.offset + self.map_offset - self.closest_map_coord_screen_coords
        )
        self.scaling = scaling

        self.offset = np.maximum(
            self.offset, window_size - 2 * self.default_offset - self.global_scaling * self.scaling * self.map_size
        )
        self.offset = np.minimum(self.offset, 0)

        self.compute_closest_map_coord_in_screen_coords(event.position().toPoint())
        self.request_repaint()

    def mousePressEvent(self, event):
        """
        Handles mouse press events on the canvas.
        """
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setFocus()
            self.panning = True
            self.last_mouse_pos = event.position().toPoint()
        elif event.button() == Qt.MouseButton.LeftButton and not self.location_transform_attributes:
            if self.selected_route is None:
                return
            self.setFocus()
            mouse_pos_screen_coords = event.position().toPoint()
            mouse_pos_screen_coords = np.array(
                [mouse_pos_screen_coords.x(), mouse_pos_screen_coords.y()], dtype="float"
            )
            mouse_pos_map_coords = self.screen_coords_to_world_coords(mouse_pos_screen_coords[None])[0]

            scenario_idx = self.selected_route.get_scenario_index_near(mouse_pos_map_coords)
            if scenario_idx is not None:
                if self.ctrl_pressed:
                    self.selected_route.remove_scenario_by_index(scenario_idx)
                    self.parent_obj.route_manager.mark_selected_route_dirty()
                    self.parent_obj.selected_scenario_ref = None
                    self.parent_obj.update_routes_list()
                    self.parent_obj.status_label.setText("Scenario removed (Ctrl+Click).")
                    self.interpolated_trace = []
                    self.request_repaint()
                    return
                self.pending_scenario_click_idx = scenario_idx
                self.pending_scenario_press_screen_pos = event.position().toPoint()
            else:
                waypoint_idx = self.selected_route.get_waypoint_index_near(mouse_pos_map_coords)
                ctrl_pressed = self.ctrl_pressed
                if waypoint_idx is not None and ctrl_pressed:
                    self.selected_route.remove_waypoint_by_index(waypoint_idx)
                    self.parent_obj.route_manager.mark_selected_route_dirty()
                    self.interpolated_trace = []
                    self.since_last_mouse_movement = time.time()
                    self.parent_obj.update_map_name_and_route_length()
                    self.parent_obj.status_label.setText("Waypoint removed (Ctrl+Click).")
                elif waypoint_idx is not None:
                    self.dragging_waypoint_idx = waypoint_idx
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    self.parent_obj.status_label.setText("Dragging waypoint...")
                else:
                    self.selected_route.add_waypoint(mouse_pos_map_coords)
                    self.parent_obj.route_manager.mark_selected_route_dirty()
                    self.interpolated_trace = []
                    self.since_last_mouse_movement = time.time()
                    self.parent_obj.update_map_name_and_route_length()
            self.request_repaint()
        elif event.button() == Qt.MouseButton.RightButton and not self.location_transform_attributes:
            if self.selected_route is None:
                return
            mouse_pos_screen_pt = event.position().toPoint()
            self.compute_closest_map_coord_in_screen_coords(mouse_pos_screen_pt)
            if self.closest_map_coord_screen_coords is not None:
                mouse_pos_screen_coords = np.array(
                    [self.closest_map_coord_screen_coords[0], self.closest_map_coord_screen_coords[1]], dtype="float"
                )
            else:
                mouse_pos_screen_coords = np.array(
                    [mouse_pos_screen_pt.x(), mouse_pos_screen_pt.y()], dtype="float"
                )
            mouse_pos_map_coords = self.screen_coords_to_world_coords(mouse_pos_screen_coords[None])[0]
            if self.selected_route.check_if_scenario_can_be_added(mouse_pos_map_coords):
                self.parent_obj.add_scenario_at_location(mouse_pos_map_coords)
            else:
                self.parent_obj.status_label.setText("Cannot add scenario here. Right-click closer to the current route.")
            self.request_repaint()
        elif event.button() == Qt.MouseButton.LeftButton:  # Add location or transform for scenario
            self.add_location_data_to_scenario(event.position().toPoint())
            self.request_repaint()

    def prepare_window_to_add_location_data_to_scenario(self):
        """
        Prepares the main window for adding location or transform data for a scenario.
        """
        if self.location_transform_attributes:
            self.parent_obj.empty_file_button.setEnabled(False)
            self.parent_obj.load_file_button.setEnabled(False)
            self.parent_obj.save_file_button.setEnabled(False)
            self.parent_obj.items_list.setEnabled(False)
            self.parent_obj.add_route_button.setEnabled(False)
            self.parent_obj.remove_route_button.setEnabled(False)

            scenario_type = self.location_transform_attributes[0]
            first_attribute = self.location_transform_attributes[1][0]
            self._set_location_pick_prompt(scenario_type, first_attribute)
            # Keep label space reserved to avoid canvas resize/jump during selection mode.

    def _set_location_pick_prompt(self, scenario_type, attribute_name):
        short_tooltip = config.get_scenario_param_tooltip(scenario_type, attribute_name)
        placement_hint = config.get_scenario_param_placement_hint(scenario_type, attribute_name)
        prompt = f"Select {attribute_name} for {scenario_type}"
        if placement_hint and placement_hint != short_tooltip:
            prompt = f"{prompt}\n{placement_hint}"
        self.parent_obj.label_add_location.setText(prompt)
        self.parent_obj.label_add_location.setToolTip(placement_hint or short_tooltip)

    def add_location_data_to_scenario(self, pos):
        """
        Adds location or transform data for a scenario based on the selected point on the canvas.
        """
        idx = 1
        while len(self.location_transform_attributes[idx]) == 3:
            idx += 1

        attr, attr_type = self.location_transform_attributes[idx]
        self.compute_closest_map_coord_in_screen_coords(pos)
        if self.closest_map_coord_screen_coords is not None:
            loc = np.array([self.closest_map_coord_screen_coords[0], self.closest_map_coord_screen_coords[1]], dtype="float")
        else:
            loc = np.array([pos.x(), pos.y()], dtype="float")
        loc = self.screen_coords_to_world_coords(loc[None])[0]
        loc = carla.Location(loc[0], loc[1])
        lane_type = carla.LaneType.Driving  # Also for 'transform' == attr_type
        if "location" in attr_type:
            if "sidewalk" in attr_type:
                lane_type = carla.LaneType.Sidewalk
            elif "bicycle" in attr_type:
                lane_type = carla.LaneType.Biking
            elif "driving" in attr_type:
                lane_type = carla.LaneType.Driving
            else:
                lane_type = carla.LaneType.Driving

        wp = self.carla_client.carla_map.get_waypoint(loc, lane_type=lane_type)
        if wp is None:
            self.parent_obj.status_label.setText("No valid waypoint here for this scenario attribute. Click a valid lane.")
            return
        wp_loc = wp.transform.location
        self.location_transform_attributes[idx].append([round(wp_loc.x, 1), round(wp_loc.y, 1), round(wp_loc.z, 1)])

        idx += 1
        if len(self.location_transform_attributes) == idx:
            self.finish_to_add_location_data_to_scenario()
        else:
            scenario_type = self.location_transform_attributes[0]
            first_attribute = self.location_transform_attributes[idx][0]
            self._set_location_pick_prompt(scenario_type, first_attribute)

    def finish_to_add_location_data_to_scenario(self):
        """
        Finalizes the process of adding location or transform data for a scenario.
        """
        self.parent_obj.empty_file_button.setEnabled(True)
        self.parent_obj.load_file_button.setEnabled(True)
        self.parent_obj.save_file_button.setEnabled(True)
        self.parent_obj.items_list.setEnabled(True)
        self.parent_obj.add_route_button.setEnabled(True)
        self.parent_obj.remove_route_button.setEnabled(True)
        self.parent_obj.label_add_location.setText("")
        self.parent_obj.label_add_location.setToolTip("")

        collected_attributes = self.location_transform_attributes[1:]
        if callable(self.location_transform_on_finish):
            self.location_transform_on_finish(collected_attributes)
        else:
            self.selected_route.add_location_transform_attributes_to_last_scenario(collected_attributes)
            self.parent_obj.route_manager.mark_selected_route_dirty()
        self.location_transform_on_finish = None
        self.location_transform_attributes.clear()
        self.request_repaint()

    def cancel_add_location_data_to_scenario(self):
        """
        Cancels location/transform point picking and returns to the scenario dialog.
        """
        if not self.location_transform_attributes:
            return

        self.parent_obj.empty_file_button.setEnabled(True)
        self.parent_obj.load_file_button.setEnabled(True)
        self.parent_obj.save_file_button.setEnabled(True)
        self.parent_obj.items_list.setEnabled(True)
        self.parent_obj.add_route_button.setEnabled(True)
        self.parent_obj.remove_route_button.setEnabled(True)
        self.parent_obj.label_add_location.setText("")
        self.parent_obj.label_add_location.setToolTip("")

        if callable(self.location_transform_on_finish):
            self.location_transform_on_finish([])
        self.location_transform_on_finish = None
        self.location_transform_attributes.clear()
        self.parent_obj.status_label.setText("Waypoint selection canceled. Returned to scenario editor.")
        self.request_repaint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.request_repaint()
        elif event.button() == Qt.MouseButton.LeftButton and self.dragging_scenario_idx is not None:
            self.dragging_scenario_idx = None
            self.pending_scenario_click_idx = None
            self.pending_scenario_press_screen_pos = None
            self.unsetCursor()
            self.parent_obj.status_label.setText("Waypoint edit mode: click to add/remove waypoints.")
            self.request_repaint()
        elif event.button() == Qt.MouseButton.LeftButton and self.dragging_waypoint_idx is not None:
            self.dragging_waypoint_idx = None
            self.unsetCursor()
            self.parent_obj.update_map_name_and_route_length()
            self.parent_obj.status_label.setText("Waypoint edit mode: click to add/remove waypoints.")
            self.request_repaint()
        elif event.button() == Qt.MouseButton.LeftButton and self.pending_scenario_click_idx is not None:
            scenario_idx = self.pending_scenario_click_idx
            self.pending_scenario_click_idx = None
            self.pending_scenario_press_screen_pos = None
            if self.selected_route is not None and scenario_idx < len(self.selected_route.scenario_types):
                self.parent_obj.on_scenario_clicked(scenario_idx)
            self.request_repaint()

    def compute_closest_map_coord_in_screen_coords(self, mouse_location_screen_coord):
        if self.min_coords is not None:
            mouse_location_screen_coord = np.array(
                [mouse_location_screen_coord.x(), mouse_location_screen_coord.y()], dtype="float"
            )
            self.closest_map_coord_screen_coords = self.get_closest_road_wp_in_screen_coord(
                mouse_location_screen_coord
            )

    def mouseMoveEvent(self, event):
        self.since_last_mouse_movement = time.time()
        self.interpolated_trace = []
        mouse_pos = event.position().toPoint()
        self.current_mouse_pos = event.position()
        self.ctrl_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self.compute_closest_map_coord_in_screen_coords(mouse_pos)
        if self.selected_route is None:
            self.refresh_hover_and_cursor()
            self.request_repaint()
            return

        if self.panning:
            diff = mouse_pos - self.last_mouse_pos
            self.offset += np.array([diff.x(), diff.y()], dtype="float")

            window_size = self.size()
            window_size = np.array([window_size.width(), window_size.height()], dtype="float")
            self.offset = np.maximum(
                self.offset, window_size - 2 * self.default_offset - self.global_scaling * self.scaling * self.map_size
            )
            self.offset = np.minimum(self.offset, 0)

            self.map_offset = np.maximum(
                0, (window_size - 2 * self.default_offset - self.scaling * self.global_scaling * self.map_size) / 2
            )

            self.last_mouse_pos = mouse_pos

        elif self.pending_scenario_click_idx is not None and not self.parent_obj.add_scenario_mode:
            press_pos = self.pending_scenario_press_screen_pos
            if press_pos is not None:
                drag_distance = np.hypot(mouse_pos.x() - press_pos.x(), mouse_pos.y() - press_pos.y())
                if drag_distance > 6.0:
                    self.dragging_scenario_idx = self.pending_scenario_click_idx
                    self.pending_scenario_click_idx = None
                    self.pending_scenario_press_screen_pos = None
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    self.parent_obj.status_label.setText("Dragging scenario...")

        elif self.dragging_waypoint_idx is not None and not self.parent_obj.add_scenario_mode:
            mouse_pos_screen_coords = np.array([mouse_pos.x(), mouse_pos.y()], dtype="float")
            mouse_pos_map_coords = self.screen_coords_to_world_coords(mouse_pos_screen_coords[None])[0]
            self.selected_route.move_waypoint_by_index(int(self.dragging_waypoint_idx), mouse_pos_map_coords)
            self.parent_obj.route_manager.mark_selected_route_dirty()
            self.interpolated_trace = []
            self.parent_obj.update_map_name_and_route_length()

        elif self.dragging_scenario_idx is not None and not self.parent_obj.add_scenario_mode:
            mouse_pos_screen_coords = np.array([mouse_pos.x(), mouse_pos.y()], dtype="float")
            mouse_pos_map_coords = self.screen_coords_to_world_coords(mouse_pos_screen_coords[None])[0]
            scenario_idx = int(self.dragging_scenario_idx)
            if scenario_idx >= len(self.selected_route.scenario_types):
                self.dragging_scenario_idx = None
                self.request_repaint()
                return
            self.selected_route.move_scenario_by_index(scenario_idx, mouse_pos_map_coords)
            self.parent_obj.route_manager.mark_selected_route_dirty()
            self.interpolated_trace = []
            self.parent_obj.update_map_name_and_route_length()

        self.refresh_hover_and_cursor()

        self.request_repaint()

    def leaveEvent(self, event):
        self.hovered_waypoint_idx = None
        self.hovered_scenario_idx = None
        self.unsetCursor()
        self.request_repaint()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.location_transform_attributes:
            self.cancel_add_location_data_to_scenario()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Control:
            self.ctrl_pressed = True
            self.current_mouse_pos = QPointF(self.mapFromGlobal(QCursor.pos()))
            self.interpolated_trace = []
            self.refresh_hover_and_cursor()
            self.request_repaint()
            self.update()
        super().keyPressEvent(event)

    def _active_location_attr_type(self):
        if not self.location_transform_attributes or len(self.location_transform_attributes) <= 1:
            return None
        idx = 1
        while idx < len(self.location_transform_attributes) and len(self.location_transform_attributes[idx]) == 3:
            idx += 1
        if idx >= len(self.location_transform_attributes):
            return None
        attr_def = self.location_transform_attributes[idx]
        if len(attr_def) < 2:
            return None
        return str(attr_def[1])

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Control:
            self.ctrl_pressed = False
            self.current_mouse_pos = QPointF(self.mapFromGlobal(QCursor.pos()))
            self.refresh_hover_and_cursor()
            self.request_repaint()
            self.update()
        super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        self.since_last_mouse_movement = time.time()
        self.interpolated_trace = []
        self.update_global_scaling(event.size())

        if self.map_size is not None:
            window_size = self.size()
            window_size = np.array([window_size.width(), window_size.height()], dtype="float")
            self.offset = np.maximum(
                self.offset, window_size - 2 * self.default_offset - self.global_scaling * self.scaling * self.map_size
            )
            self.offset = np.minimum(self.offset, 0)

            self.map_offset = np.maximum(
                0, (window_size - 2 * self.default_offset - self.scaling * self.global_scaling * self.map_size) / 2
            )

        self.compute_closest_map_coord_in_screen_coords(self.last_mouse_pos)
        self.request_repaint()

    def update_global_scaling(self, widget_size):
        if self.map_width is not None:
            window_size = self.size()
            window_size = np.array([window_size.width(), window_size.height()], dtype="float")
            self.global_scaling = (
                (window_size - 2 * self.default_offset) / np.array([self.map_width, self.map_height])
            ).min()

            self.map_offset = np.maximum(
                0, (window_size - 2 * self.default_offset - self.scaling * self.global_scaling * self.map_size) / 2
            )
            self.request_repaint()

    # get the closest world coordinate
    def get_closest_road_wp_in_screen_coord(self, mouse_location_screen_coord):
        mouse_location_world_coord = self.screen_coords_to_world_coords(mouse_location_screen_coord[None])[0]
        active_attr_type = self._active_location_attr_type()
        use_biking = active_attr_type is not None and "bicycle" in active_attr_type
        lane_type = carla.LaneType.Biking if use_biking else (carla.LaneType.Driving | carla.LaneType.Parking)

        wp = None
        if self.carla_client is not None and self.carla_client.carla_map is not None:
            wp = self.carla_client.carla_map.get_waypoint(
                carla.Location(float(mouse_location_world_coord[0]), float(mouse_location_world_coord[1])),
                lane_type=lane_type,
            )

        if wp is not None:
            closest_map_wp_world_coord = np.array([wp.transform.location.x, wp.transform.location.y], dtype="float")
        else:
            use_tree_bike = use_biking and self.biking_tree is not None
            query_tree = self.biking_tree if use_tree_bike else self.tree
            query_points = self.biking_waypoints_np if use_tree_bike else self.all_waypoints_np
            _, idx = query_tree.query(mouse_location_world_coord)
            closest_map_wp_world_coord = query_points[idx]
        closest_map_wp_screen_coord = self.world_coords_to_screen_coords(closest_map_wp_world_coord[None])[0]

        return closest_map_wp_screen_coord

    def world_coords_to_screen_coords(self, world_coords):
        # transforms carla world coordinates to screen coordinates
        # shape: [N, 2]: np.array
        screen_coords = self.global_scaling * self.scaling * (world_coords - self.min_coords[None, :])
        screen_coords += self.default_offset
        screen_coords += self.offset[None, :]
        screen_coords += self.map_offset[None, :]

        return screen_coords

    def screen_coords_to_world_coords(self, screen_coords):
        # transforms screen coordinates to carla world coordinates
        # shape: [N, 2]: np.array

        world_coords = screen_coords - self.default_offset - self.offset[None, :] - self.map_offset[None, :]
        world_coords /= self.global_scaling
        world_coords /= self.scaling
        world_coords += self.min_coords[None, :]

        return world_coords

    def select_coords_inside_window(self, screen_coords):
        window_size = self.size()
        window_size = np.array([window_size.width(), window_size.height()], dtype="float")

        flag = (
            (screen_coords[:, 0] >= 0)
            & (screen_coords[:, 1] >= 0)
            & (screen_coords[:, 0] <= window_size[0])
            & (screen_coords[:, 1] <= window_size[1])
        )

        filtered_screen_coords = screen_coords[flag]

        return filtered_screen_coords


class Window(QWidget):
    def __init__(self, carla_client, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Route Editor")
        self.resize(1280, 780)
        self.center()

        self.carla_client = carla_client
        self.route_manager = RouteManager(carla_client)
        self.add_scenario_mode = False
        self.selected_scenario_ref = None  # tuple: (route_id, scenario_idx)
        self._suppress_tree_callbacks = False
        self.last_save_directory = None
        self.autosave_dir = Path(".autosave")
        self.autosave_dir.mkdir(parents=True, exist_ok=True)

        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        canvas_layout = QVBoxLayout()

        session_button_row = QHBoxLayout()
        self.new_session_button = QPushButton("New Session")
        self.new_session_button.clicked.connect(self.on_new_session_button_click)
        session_button_row.addWidget(self.new_session_button)
        controls_layout.addLayout(session_button_row)

        load_save_row_1 = QHBoxLayout()
        self.load_route_button = QPushButton("Load Route File")
        self.load_route_button.clicked.connect(self.on_load_route_button_click)
        load_save_row_1.addWidget(self.load_route_button)
        self.load_project_button = QPushButton("Load Route Folder")
        self.load_project_button.clicked.connect(self.on_load_project_button_click)
        load_save_row_1.addWidget(self.load_project_button)
        controls_layout.addLayout(load_save_row_1)

        load_save_row_2 = QHBoxLayout()
        self.save_route_button = QPushButton("Save Route")
        self.save_route_button.clicked.connect(self.on_save_route_button_click)
        self.save_route_button.setEnabled(False)
        load_save_row_2.addWidget(self.save_route_button)
        self.save_route_as_button = QPushButton("Save Route As")
        self.save_route_as_button.clicked.connect(self.on_save_route_as_button_click)
        self.save_route_as_button.setEnabled(False)
        load_save_row_2.addWidget(self.save_route_as_button)
        controls_layout.addLayout(load_save_row_2)

        self.save_all_button = QPushButton("Save All Routes To Folder")
        self.save_all_button.clicked.connect(self.on_save_all_button_click)
        self.save_all_button.setEnabled(False)
        controls_layout.addWidget(self.save_all_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setFixedHeight(10)
        controls_layout.addWidget(separator)

        route_button_row = QHBoxLayout()
        self.add_route_button = QPushButton("Add Route")
        self.add_route_button.clicked.connect(self.on_add_route_button_click)
        self.add_route_button.setEnabled(False)
        route_button_row.addWidget(self.add_route_button)

        self.remove_route_button = QPushButton("Remove Route")
        self.remove_route_button.clicked.connect(self.on_remove_route_button_click)
        self.remove_route_button.setEnabled(False)
        route_button_row.addWidget(self.remove_route_button)
        controls_layout.addLayout(route_button_row)

        self.label_selected_town = QLabel("No town selected")
        self.label_selected_town.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        controls_layout.addWidget(self.label_selected_town)

        self.route_edit_hint = QLabel(
            "Right click near route: add scenario | Left drag waypoint/scenario: move | Ctrl+Left click: remove"
        )
        self.route_edit_hint.setWordWrap(True)
        controls_layout.addWidget(self.route_edit_hint)

        self.show_other_routes_checkbox = QCheckBox("Show other routes")
        self.show_other_routes_checkbox.setChecked(False)
        self.show_other_routes_checkbox.toggled.connect(lambda _: self.canvas.request_repaint())
        controls_layout.addWidget(self.show_other_routes_checkbox)

        scenario_row = QHBoxLayout()
        controls_layout.addLayout(scenario_row)

        self.items_list = QTreeWidget()
        self.items_list.setColumnCount(1)
        self.items_list.setHeaderHidden(True)
        self.items_list.setUniformRowHeights(True)
        self.items_list.setIndentation(16)
        self.items_list.itemClicked.connect(self.on_list_item_clicked)
        self.items_list.currentItemChanged.connect(self.on_current_item_changed)
        controls_layout.addWidget(self.items_list)

        font = self.items_list.font()
        font.setPointSize(12)
        self.items_list.setFont(font)

        self.label_add_location = QLabel()
        font2 = self.label_add_location.font()
        self.label_add_location.setStyleSheet("color: #ce3c3c; font-weight: 700;")
        font2.setPointSize(14)
        self.label_add_location.setFont(font2)
        self.label_add_location.setWordWrap(True)
        self.label_add_location.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.label_add_location.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        canvas_layout.addWidget(self.label_add_location)
        self.label_add_location.setText("")

        self.canvas = Canvas(self.carla_client, parent=self, route_manager=self.route_manager)
        self.canvas.setEnabled(False)
        self.canvas.setMinimumWidth(700)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas_layout.addWidget(self.canvas)

        self.status_label = QLabel("Start a new session or load routes.")
        self.status_label.setStyleSheet("color: #5d6a75;")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        canvas_layout.addWidget(self.status_label)
        canvas_layout.setStretch(1, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_pane = QWidget()
        left_pane.setLayout(controls_layout)
        right_pane = QWidget()
        right_pane.setLayout(canvas_layout)
        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.setSizes([340, 900])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.apply_styles()

        # Backwards-compatible names used by Canvas while selecting scenario location/transform attributes.
        self.empty_file_button = self.new_session_button
        self.load_file_button = self.load_route_button
        self.save_file_button = self.save_route_button

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave_dirty_routes)
        self.autosave_timer.start(60_000)

    def apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                font-size: 13px;
                color: #1f2a33;
                background: #f4f7fb;
            }
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
                border: 1px solid #c8d4e2;
                background: #ffffff;
            }
            QPushButton:checked {
                background: #d9ecff;
                border-color: #5a9ad6;
            }
            QPushButton:disabled {
                color: #9ba7b4;
                background: #eef2f7;
            }
            QTreeWidget {
                border: 1px solid #d1d7dd;
                border-radius: 6px;
                padding: 4px;
                background: #ffffff;
            }
            QLabel {
                background: transparent;
            }
            """
        )

    def build_initial_values_for_scenario_dialog(self, scenario_elem, scenario_type):
        initial_values = {}
        for attr_def in config.SCENARIO_TYPES.get(scenario_type, []):
            attr_name, attr_type = attr_def[0], attr_def[1]
            elem = scenario_elem.find(attr_name)
            if elem is None:
                continue
            if attr_type in ("value", "bool"):
                value = elem.get("value")
                if value is not None:
                    if scenario_type in ("PermutedConstructionObstacle", "PermutedConstructionObstacleTwoWays") and attr_name == "cones":
                        initial_values[attr_name] = str(value)
                    else:
                        initial_values[attr_name] = self._parse_scalar(value)
            elif attr_type == "interval":
                from_value = elem.get("from")
                to_value = elem.get("to")
                if from_value is not None and to_value is not None:
                    initial_values[attr_name] = [self._parse_scalar(from_value), self._parse_scalar(to_value)]
            elif attr_type == "choice":
                value = elem.get("value")
                if value is not None:
                    initial_values[attr_name] = value
        return initial_values

    @staticmethod
    def _parse_scalar(value):
        value = str(value).strip()
        try:
            if "." in value:
                as_float = float(value)
                return int(as_float) if as_float.is_integer() else as_float
            return int(value)
        except ValueError:
            try:
                as_float = float(value)
                return int(as_float) if as_float.is_integer() else as_float
            except ValueError:
                return value

    def build_initial_values_for_graphical_editor_dialog(self, scenario_elem, scenario_type):
        initial_values = self.build_initial_values_for_scenario_dialog(scenario_elem, scenario_type)

        baseline_overlay_elem = scenario_elem.find("baseline_overlay")
        if baseline_overlay_elem is not None:
            baseline_overlay = baseline_overlay_elem.get("value")
            if baseline_overlay is not None:
                initial_values["baseline_overlay"] = baseline_overlay

        overlay_direction_elem = scenario_elem.find("overlay_direction")
        if overlay_direction_elem is not None:
            overlay_direction = overlay_direction_elem.get("value")
            if overlay_direction is not None:
                initial_values["overlay_direction"] = overlay_direction

        objects_elem = scenario_elem.find("objects")
        if objects_elem is not None:
            initial_values["objects"] = dict(objects_elem.attrib)

        return initial_values

    def pick_single_location_attribute(self, scenario_type, attribute, attr_type):
        result = {"value": None}
        loop = QEventLoop(self)

        self.canvas.location_transform_attributes = [scenario_type, [attribute, attr_type]]

        def _finish_location_pick(collected_attributes):
            if collected_attributes:
                _, _, attr_value = collected_attributes[0]
                result["value"] = list(attr_value)
            loop.quit()

        self.canvas.location_transform_on_finish = _finish_location_pick
        self.canvas.prepare_window_to_add_location_data_to_scenario()
        self.status_label.setText(f"Select {attribute} for {scenario_type}.")
        loop.exec()
        return result["value"]

    def open_scenario_editor(self, scenario_type, initial_values=None):
        profile = get_scenario_editor_profile(scenario_type)
        if profile.kind == "none":
            return [], False

        if profile.kind == "graphical":
            is_permuted_construction = scenario_type in ("PermutedConstructionObstacle", "PermutedConstructionObstacleTwoWays")
            is_bad_parking = scenario_type in ("BadParkingObstacle", "BadParkingObstacleTwoWays")
            is_road_blocked = scenario_type == "RoadBlocked"
            effective_initial_values = dict(initial_values or {})
            layout_attributes = []

            def _apply_attributes_to_initial_values(attrs):
                for attr_name, attr_type, attr_value in attrs:
                    if attr_type in ("value", "bool", "choice"):
                        effective_initial_values[attr_name] = attr_value
                    elif attr_type == "interval":
                        effective_initial_values[attr_name] = list(attr_value)
                    elif attr_type == "objects":
                        effective_initial_values[attr_name] = dict(attr_value)

            while True:
                param_dialog = ScenarioAttributeDialog(
                    scenario_type,
                    parent=self,
                    initial_values=effective_initial_values,
                    allow_layout_customization=True,
                )
                param_dialog.exec()
                if param_dialog.was_cancelled:
                    return [], True
                scenario_attributes = [list(x) for x in param_dialog.scenario_attributes]
                _apply_attributes_to_initial_values(scenario_attributes)

                if not param_dialog.request_layout_customization:
                    final_attributes = [list(x) for x in scenario_attributes]
                    final_attributes.extend([list(x) for x in layout_attributes])
                    return final_attributes, False

                if is_permuted_construction:
                    dialog = PermutedConstructionLayoutDialog(
                        scenario_type,
                        parent=self,
                        initial_values=effective_initial_values,
                        base_attributes=scenario_attributes,
                    )
                elif is_bad_parking:
                    dialog = BadParkingLayoutDialog(
                        scenario_type,
                        parent=self,
                        initial_values=effective_initial_values,
                        base_attributes=scenario_attributes,
                    )
                elif is_road_blocked:
                    dialog = RoadBlockedLayoutDialog(
                        scenario_type,
                        parent=self,
                        initial_values=effective_initial_values,
                        base_attributes=scenario_attributes,
                    )
                else:
                    dialog = CustomObstacleDialog(
                        scenario_type,
                        parent=self,
                        initial_values=effective_initial_values,
                        editor_profile={
                            "object_mode": profile.object_mode,
                            "overlay_types": list(profile.overlay_types),
                        },
                        base_attributes=scenario_attributes,
                    )
                dialog.exec()
                if dialog.was_cancelled:
                    continue

                dialog_attributes = [list(x) for x in dialog.scenario_attributes]
                _apply_attributes_to_initial_values(dialog_attributes)
                if is_bad_parking:
                    layout_attributes = [x for x in dialog_attributes if x[0] in ("vehicle", "x", "y", "yaw")]
                elif is_permuted_construction or is_road_blocked:
                    layout_attributes = []
                else:
                    layout_attributes = [x for x in dialog_attributes if x[0] in ("baseline_overlay", "overlay_direction", "objects")]
                # Return to parameter dialog after saving in customizer.
                continue

        effective_initial_values = dict(initial_values or {})
        while True:
            dialog = ScenarioAttributeDialog(scenario_type, parent=self, initial_values=effective_initial_values)
            dialog.exec()
            if dialog.was_cancelled:
                return [], True
            if dialog.requested_map_pick is not None:
                effective_initial_values = dict(dialog.requested_map_pick_initial_values or effective_initial_values)
                attr_name, attr_type = dialog.requested_map_pick
                picked_value = self.pick_single_location_attribute(scenario_type, attr_name, attr_type)
                if picked_value is not None:
                    effective_initial_values[attr_name] = picked_value
                continue
            return [list(x) for x in dialog.scenario_attributes], False

    def add_scenario_at_location(self, mouse_pos_map_coords):
        if not self.route_manager.routes:
            return

        selected_route = self.route_manager.routes[self.route_manager.selected_route_id]
        if not selected_route.check_if_scenario_can_be_added(mouse_pos_map_coords):
            self.status_label.setText("Cannot add scenario here. Click closer to the current route.")
            return

        scenario_selection_dialog = ScenarioSelectionDialog()
        selected_scenario = scenario_selection_dialog.selected_scenario
        if selected_scenario is None:
            return

        scenario_attributes = []
        try:
            _, snapped_loc = selected_route._get_snapped_driving_or_parking_wp_loc(mouse_pos_map_coords)
            self.canvas.location_transform_anchor = [round(snapped_loc[0], 1), round(snapped_loc[1], 1), round(snapped_loc[2], 1)]
        except Exception:
            self.canvas.location_transform_anchor = list(mouse_pos_map_coords)
        self.canvas.request_repaint()
        scenario_attributes, was_cancelled = self.open_scenario_editor(selected_scenario)
        self.canvas.location_transform_anchor = None
        self.canvas.request_repaint()
        if was_cancelled:
            return

        selected_route.add_scenario(mouse_pos_map_coords, selected_scenario, scenario_attributes)
        self.route_manager.mark_selected_route_dirty()
        self.canvas.interpolated_trace = []
        self.canvas.since_last_mouse_movement = time.time()
        self.canvas.request_repaint()
        self.update_map_name_and_route_length()
        self.update_routes_list()
        self.status_label.setText("Waypoint edit mode: click to add/remove waypoints.")

    def on_scenario_clicked(self, scenario_idx):
        if self.route_manager.selected_route_id is None:
            return
        route_id = self.route_manager.selected_route_id
        self.selected_scenario_ref = (route_id, int(scenario_idx))
        item = self.find_tree_item_for_scenario(route_id, int(scenario_idx))
        if item is not None:
            parent = item.parent()
            if parent is not None:
                parent.setExpanded(True)
            self._suppress_tree_callbacks = True
            self.items_list.setCurrentItem(item)
            self.items_list.scrollToItem(item)
            self._suppress_tree_callbacks = False
        self.canvas.request_repaint()
        self.status_label.setText(f"Selected scenario {scenario_idx} on route {route_id}.")
        self.on_edit_selected_scenario(route_id, int(scenario_idx))

    def on_edit_selected_scenario(self, route_id, scenario_idx):
        if route_id not in self.route_manager.routes:
            return
        selected_route = self.route_manager.routes[route_id]
        if scenario_idx is None or scenario_idx >= len(selected_route.scenarios):
            return
        if not self.scenario_has_editable_attributes(route_id, scenario_idx):
            self.status_label.setText("This scenario has no editable attributes.")
            return

        scenario_type = selected_route.scenario_types[scenario_idx]
        scenario_elem = selected_route.scenarios[scenario_idx]
        if scenario_idx < len(selected_route.scenario_trigger_points):
            self.canvas.location_transform_anchor = list(selected_route.scenario_trigger_points[scenario_idx])
        else:
            self.canvas.location_transform_anchor = None
        self.canvas.request_repaint()
        profile = get_scenario_editor_profile(scenario_type)
        if profile.kind == "graphical":
            initial_values = self.build_initial_values_for_graphical_editor_dialog(scenario_elem, scenario_type)
            scenario_attributes, was_cancelled = self.open_scenario_editor(scenario_type, initial_values=initial_values)
            if was_cancelled:
                self.canvas.location_transform_anchor = None
                self.canvas.request_repaint()
                return
            selected_route.update_scenario_by_index(scenario_idx, scenario_attributes)
        else:
            initial_values = self.build_initial_values_for_scenario_dialog(scenario_elem, scenario_type)
            scenario_attributes, was_cancelled = self.open_scenario_editor(scenario_type, initial_values=initial_values)
            if was_cancelled:
                self.canvas.location_transform_anchor = None
                self.canvas.request_repaint()
                return

            selected_route.update_scenario_by_index(scenario_idx, scenario_attributes)
        self.canvas.location_transform_anchor = None
        self.route_manager.mark_selected_route_dirty()
        self.canvas.request_repaint()
        self.update_routes_list()
        self.status_label.setText("Scenario updated.")

    def find_tree_item_for_scenario(self, route_id, scenario_idx):
        for i in range(self.items_list.topLevelItemCount()):
            top_item = self.items_list.topLevelItem(i)
            data = top_item.data(0, Qt.ItemDataRole.UserRole)
            if not (isinstance(data, tuple) and data[0] == "route" and data[1] == route_id):
                continue
            for j in range(top_item.childCount()):
                child = top_item.child(j)
                child_data = child.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(child_data, tuple) and child_data == ("scenario", route_id, scenario_idx):
                    return child
        return None

    def _clear_selected_scenario(self):
        self.selected_scenario_ref = None

    def scenario_has_editable_attributes(self, route_id, scenario_idx):
        if route_id not in self.route_manager.routes:
            return False
        route = self.route_manager.routes[route_id]
        if scenario_idx < 0 or scenario_idx >= len(route.scenario_types):
            return False
        scenario_type = route.scenario_types[scenario_idx]
        profile = get_scenario_editor_profile(scenario_type)
        if profile.kind == "graphical":
            return True
        if profile.kind == "none":
            return False
        fields = config.SCENARIO_TYPES.get(scenario_type, [])
        for field in fields:
            attr_type = field[1]
            if attr_type in ("value", "bool", "interval", "choice", "transform") or "location" in attr_type:
                return True
        return False

    def show_yes_no_dialog(self, text):
        reply = QMessageBox.question(
            self,
            "Confirmation",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return reply == QMessageBox.StandardButton.Yes

    def center(self):
        frame_geometry = self.frameGeometry()
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            center_point = screen.availableGeometry().center()
            frame_geometry.moveCenter(center_point)
            self.move(frame_geometry.topLeft())

    def on_add_route_button_click(self):
        self.route_manager.add_empty_route()
        self.canvas.setEnabled(True)
        self.update_routes_list()
        self.update_map_name_and_route_length()
        self.status_label.setText("Added a new route. Waypoint edit mode: click to add/remove waypoints.")

    def on_remove_route_button_click(self):
        if len(self.route_manager.routes) > 1:
            self.route_manager.remove_selected_route()
            self.update_routes_list()
            self.update_map_name_and_route_length()
            self.status_label.setText("Removed selected route.")
        elif len(self.route_manager.routes) == 1:
            self.route_manager.remove_selected_route()
            self.update_routes_list()
            self.update_map_name_and_route_length()
            self.canvas.setEnabled(False)
            self.status_label.setText("Removed last route.")

    def on_new_session_button_click(self):
        create_empty_file = True
        if self.route_manager.routes:
            create_empty_file = self.show_yes_no_dialog(
                "Are you sure you want to discard the current routes without saving?"
            )

        if create_empty_file:
            map_selection_window = MapSelectionDialog(self.carla_client, self)

            map_name = map_selection_window.selected_map_name
            if map_name is not None:
                LoadingIndicatorWindow(None, "Loading map...", lambda: self.route_manager.empty_routes(map_name))

                self.update_map_name_and_route_length()
                self.add_route_button.setEnabled(True)
                self.save_route_button.setEnabled(True)
                self.save_route_as_button.setEnabled(True)
                self.save_all_button.setEnabled(True)
                self.remove_route_button.setEnabled(True)
                self.canvas.setEnabled(True)
                self.update_routes_list()
                self.canvas.reset_map_offset_and_scaling()
                self.status_label.setText(f"Loaded map {map_name}.")

    def update_routes_list(self):
        routes, selected_route_id = self.route_manager.routes, self.route_manager.selected_route_id
        expanded_route_ids = set()
        current_item = self.items_list.currentItem()
        current_selection = current_item.data(0, Qt.ItemDataRole.UserRole) if current_item is not None else None
        for i in range(self.items_list.topLevelItemCount()):
            top_item = self.items_list.topLevelItem(i)
            data = top_item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and data[0] == "route" and top_item.isExpanded():
                expanded_route_ids.add(data[1])

        self.items_list.clear()
        target_item_to_select = None

        for route_id in sorted(routes.keys()):
            route_path = self.route_manager.route_file_paths.get(route_id)
            dirty = route_id in self.route_manager.dirty_route_ids
            source_name = Path(route_path).name if route_path else "unsaved"
            dirty_prefix = "* " if dirty else ""
            route = routes[route_id]

            route_item = QTreeWidgetItem([f"{dirty_prefix}Route {route_id} [{source_name}]"])
            route_item.setData(0, Qt.ItemDataRole.UserRole, ("route", route_id))
            route_item.setTextAlignment(0, Qt.AlignmentFlag.AlignHCenter)
            self.items_list.addTopLevelItem(route_item)

            for scenario_idx, scenario_type in enumerate(route.scenario_types):
                scenario_item = QTreeWidgetItem([f"Scenario {scenario_idx}: {scenario_type}"])
                scenario_item.setData(0, Qt.ItemDataRole.UserRole, ("scenario", route_id, scenario_idx))
                route_item.addChild(scenario_item)
                if current_selection == ("scenario", route_id, scenario_idx):
                    target_item_to_select = scenario_item

            if route_id in expanded_route_ids:
                route_item.setExpanded(True)
            if current_selection == ("route", route_id):
                target_item_to_select = route_item
            elif route_id == selected_route_id and target_item_to_select is None:
                target_item_to_select = route_item

        if target_item_to_select is not None:
            self._suppress_tree_callbacks = True
            self.items_list.setCurrentItem(target_item_to_select)
            self.items_list.scrollToItem(target_item_to_select)
            self._suppress_tree_callbacks = False

        if self.route_manager.selected_route_id is not None:
            selected_route = self.route_manager.routes[self.route_manager.selected_route_id]
            self.canvas.update_selected_route(selected_route)
            self.canvas.update_data_from_carla_client()
        else:
            self.canvas.update_selected_route(None)
            self.canvas.request_repaint()

    def closeEvent(self, event):
        if event.spontaneous():
            if self.route_manager.dirty_route_ids:
                close_window = self.show_yes_no_dialog(
                    "You have unsaved route changes. Close anyway?"
                )
                if not close_window:
                    event.ignore()
                    return

        event.accept()

    def on_load_route_button_click(self):
        options = QFileDialog.Option.DontUseCustomDirectoryIcons
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Route XML", "", "XML Files (*.xml)", options=options)
        if file_name:
            append = False
            if self.route_manager.routes:
                append = self.show_yes_no_dialog("Append this route file to current session?")
                if not append and self.route_manager.dirty_route_ids:
                    overwrite = self.show_yes_no_dialog("Discard current unsaved routes and load the selected file?")
                    if not overwrite:
                        return
            LoadingIndicatorWindow(None, "Loading route file...", lambda: self.route_manager.load_routes_from_file(file_name, append=append))
            self.update_map_name_and_route_length()
            self.add_route_button.setEnabled(True)
            self.save_route_button.setEnabled(True)
            self.save_route_as_button.setEnabled(True)
            self.save_all_button.setEnabled(True)
            self.remove_route_button.setEnabled(True)
            self.canvas.setEnabled(True)
            self.update_routes_list()
            self.canvas.reset_map_offset_and_scaling()
            self.status_label.setText(f"Loaded route file: {Path(file_name).name}")

    def on_load_project_button_click(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Route Folder")
        if directory:
            LoadingIndicatorWindow(None, "Loading route folder...", lambda: self.route_manager.load_routes_from_directory(directory))
            if self.route_manager.routes:
                self.update_map_name_and_route_length()
                self.add_route_button.setEnabled(True)
                self.save_route_button.setEnabled(True)
                self.save_route_as_button.setEnabled(True)
                self.save_all_button.setEnabled(True)
                self.remove_route_button.setEnabled(True)
                self.canvas.setEnabled(True)
                self.update_routes_list()
                self.canvas.reset_map_offset_and_scaling()
                self.last_save_directory = directory
                self.status_label.setText(f"Loaded route folder: {directory}")

    def on_save_route_button_click(self):
        save_result = self.route_manager.save_selected_route_to_file()
        if save_result is None:
            self.on_save_route_as_button_click()
            return
        save_path, baseline_path = save_result
        self.update_routes_list()
        if baseline_path is None:
            self.status_label.setText(f"Saved route to {save_path}")
        else:
            self.status_label.setText(f"Saved route to {save_path} and baseline to {baseline_path}")

    def on_save_route_as_button_click(self):
        options = QFileDialog.Option.DontUseCustomDirectoryIcons
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Route As", "", "XML Files (*.xml)", options=options)
        if file_name:
            save_path, baseline_path = self.route_manager.save_selected_route_to_file(file_name)
            self.update_routes_list()
            if baseline_path is None:
                self.status_label.setText(f"Saved route to {save_path}")
            else:
                self.status_label.setText(f"Saved route to {save_path} and baseline to {baseline_path}")

    def on_save_all_button_click(self):
        directory = QFileDialog.getExistingDirectory(self, "Save All Routes To Folder")
        if directory:
            self.route_manager.save_all_routes_to_directory(directory)
            self.last_save_directory = directory
            self.update_routes_list()
            self.status_label.setText(f"Saved all routes to {directory}")

    def autosave_dirty_routes(self):
        if self.route_manager.dirty_route_ids:
            self.route_manager.save_dirty_routes_to_directory(str(self.autosave_dir))
            self.status_label.setText(f"Autosaved dirty routes to {self.autosave_dir}")

    def update_map_name_and_route_length(self):
        selected_route_id = self.route_manager.selected_route_id
        if selected_route_id is None or selected_route_id not in self.route_manager.routes:
            self.label_selected_town.setText("No route selected")
            return

        selected_route = self.route_manager.routes[selected_route_id]
        self.label_selected_town.setText(f"{selected_route.map_name} - {round(selected_route.route_length/1000.,3)} km")

    def on_list_item_clicked(self, item):
        if self._suppress_tree_callbacks:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        if isinstance(data, tuple) and data[0] == "route":
            route_id = data[1]
            self.route_manager.selected_route_id = int(route_id)
            selected_route = self.route_manager.routes[self.route_manager.selected_route_id]
            self.canvas.update_selected_route(selected_route)
            self._clear_selected_scenario()
            self.update_map_name_and_route_length()
        elif isinstance(data, tuple) and data[0] == "scenario":
            _, route_id, scenario_idx = data
            self.route_manager.selected_route_id = int(route_id)
            selected_route = self.route_manager.routes[self.route_manager.selected_route_id]
            self.canvas.update_selected_route(selected_route)
            self.update_map_name_and_route_length()
            self.on_scenario_clicked(int(scenario_idx))

    def on_current_item_changed(self, current, previous):
        if self._suppress_tree_callbacks:
            return
        if current is not None:
            self.on_list_item_clicked(current)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-data-dir", type=str, default="carla_map_data", help="The path of the directory with the map data"
    )
    parser.add_argument(
        "--xodr-map-dir", type=str, default="carla_xodr", help="The path of the directory with OpenDRIVE maps"
    )
    args = parser.parse_args()

    carla_client = CarlaClient(args.map_data_dir, args.xodr_map_dir)

    app = QApplication(sys.argv)
    main_window = Window(carla_client)
    main_window.show()
    sys.exit(app.exec())


