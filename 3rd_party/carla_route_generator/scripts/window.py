"""
Route Creation and Scenario Editing Tool for CARLA Simulator

This Python script provides a graphical user interface (GUI) for creating and editing routes,
as well as defining scenarios, within the CARLA driving simulator environment. It allows users
to load, save, and manipulate routes by adding or removing waypoints, defining scenario trigger
points, and specifying scenario attributes. The tool also supports visualizing the map, routes,
and associated elements such as stop signs and traffic lights.
"""

import sys
from PyQt5.QtWidgets import (
    QLabel,
    QMessageBox,
    QFrame,
    QFileDialog,
    QListWidgetItem,
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
)
from PyQt5.QtCore import Qt, QTimer, QPointF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap, QColor
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


class Separator(QWidget):
    """
    A simple horizontal separator widget.
    """

    def __init__(self, separator_height=2):
        super().__init__()
        self.setFixedHeight(separator_height)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


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

        self.scaling = 1.0
        self.global_scaling = 1.0
        self.offset = np.array([0.0, 0.0])
        self.default_offset = default_offset
        self.map_offset = np.array([0.0, 0.0])
        self.panning = False
        self.since_last_mouse_movement = time.time()
        self.location_transform_attributes = []

        # Load the icons for stop signs and traffic lights
        self.stop_sign_pixmap = QPixmap("scripts/images/stop_sign_icon.png")
        self.traffic_light_pixmap = QPixmap("scripts/images/traffic_light_icon.png")

        self.road_waypoints_np = None
        self.parking_waypoints_np = None
        self.traffic_light_centers_np = None
        self.stop_sign_centers_np = None
        self.min_coords = None
        self.map_width = None
        self.map_height = None
        self.map_size = None
        self.closest_map_coord_screen_coords = None
        self.selected_route = None
        self.last_mouse_pos = QPointF(0, 0)

        # Define colors
        self.STOP_SIGN_COLOR = QColor(180, 50, 50)
        self.TRAFFIC_LIGHT_COLOR = QColor(50, 180, 50)
        self.BACKGROUND_COLOR = QColor(255, 255, 255)
        self.MAP_COLOR = QColor(0, 0, 0)
        self.PARKING_LOT_COLOR = QColor(100, 100, 255)
        self.NEW_ROUTE_PART_COLOR = QColor(130, 250, 130)
        self.ROUTE_COLOR = QColor(0, 170, 0)
        self.OTHER_ROUTES_COLOR = QColor(170, 220, 170)
        self.FIRST_WP = QColor(0, 255, 0)
        self.CURSOR_COLOR = QColor(0, 255, 0)
        self.SCENARIO_COLOR = QColor(255, 100, 100)

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
        self.SCALING_SCENARIO_TEXTS = 16.0

        self.timer = QTimer(self)  # Create a QTimer object
        self.timer.timeout.connect(
            self.update_when_no_movement
        )  # Connect timeout signal to update_when_no_movement method
        self.timer.start(int(self.fps_inv * 1000))  # Start the timer with a timeout based on the FPS

    def update_when_no_movement(self):
        """
        Updates the canvas when there is no mouse movement for a certain period of time.
        This method interpolates the route between the last waypoint and the closest map coordinate.
        """
        if (
            not self.location_transform_attributes
            and not self.interpolated_trace
            and self.closest_map_coord_screen_coords is not None
            and time.time() - self.since_last_mouse_movement > self.interpolating_after_ticks_of_no_mouse_movement
        ):
            closest_map_loc = self.screen_coords_to_world_coords(self.closest_map_coord_screen_coords[None])[0]
            closest_map_loc = carla.Location(closest_map_loc[0], closest_map_loc[1])
            self.interpolated_trace = self.selected_route.interpolate_from_last_wp(closest_map_loc)

        current_time = time.time()
        if current_time - self.last_screen_update > self.fps_inv:
            self.last_screen_update = current_time
            self.update()

    def update_data_from_carla_client(self):
        """
        Updates the map data from the CARLA client, including waypoints, traffic light centers, stop sign centers, and map dimensions.
        """
        self.road_waypoints_np = self.carla_client.road_waypoints_np
        self.parking_waypoints_np = self.carla_client.parking_waypoints_np
        self.traffic_light_centers_np = self.carla_client.traffic_light_centers_np
        self.stop_sign_centers_np = self.carla_client.stop_sign_centers_np
        self.min_coords = self.carla_client.min_coords
        self.map_width = self.carla_client.map_width
        self.map_height = self.carla_client.map_height
        self.map_size = self.carla_client.map_size
        self.interpolated_trace = []

        self.all_waypoints_np = np.concatenate([self.road_waypoints_np, self.parking_waypoints_np], axis=0)
        self.tree = cKDTree(self.all_waypoints_np)

        self.closest_map_coord_screen_coords = self.road_waypoints_np[0, :2]

    def update_selected_route(self, selected_route):
        """
        Updates the currently selected route and resets the canvas view.
        """
        self.selected_route = selected_route

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
            painter.setRenderHint(QPainter.Antialiasing)

            road_waypoints = self.world_coords_to_screen_coords(self.road_waypoints_np[:, :2])
            road_waypoints = self.select_coords_inside_window(road_waypoints)
            parking_waypoints = self.world_coords_to_screen_coords(self.parking_waypoints_np[:, :2])
            parking_waypoints = self.select_coords_inside_window(parking_waypoints)
            sparse_waypoints = np.array(self.selected_route.waypoints).reshape((-1, 3))[:, :2]
            sparse_waypoints = self.world_coords_to_screen_coords(sparse_waypoints)
            sparse_waypoints = self.select_coords_inside_window(sparse_waypoints)
            scenario_trigger_points = np.array(self.selected_route.scenario_trigger_points).reshape((-1, 3))[:, :2]
            scenario_trigger_points = self.world_coords_to_screen_coords(scenario_trigger_points)
            dense_waypoints = np.array(self.selected_route.dense_waypoints).reshape((-1, 3))[:, :2]
            dense_waypoints = self.world_coords_to_screen_coords(dense_waypoints)
            dense_waypoints = self.select_coords_inside_window(dense_waypoints)
            interpolated_trace = np.array(self.interpolated_trace).reshape((-1, 3))[:, :2]
            interpolated_trace = self.world_coords_to_screen_coords(interpolated_trace)
            interpolated_trace = self.select_coords_inside_window(interpolated_trace)

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
                    + dense_waypoints.shape[0]
                    + interpolated_trace.shape[0]
                    + other_routes_dense_points.shape[0]
                )
                // self.max_drawn_points,
            )
            road_waypoints = road_waypoints[::n_skip]
            parking_waypoints = parking_waypoints[::n_skip]
            dense_waypoints = dense_waypoints[::n_skip]
            interpolated_trace = interpolated_trace[::n_skip]
            other_routes_dense_points = other_routes_dense_points[::n_skip]

            road_waypoints = [QPointF(x, y) for (x, y) in road_waypoints.tolist()]
            parking_waypoints = [QPointF(x, y) for (x, y) in parking_waypoints.tolist()]
            sparse_waypoints = [QPointF(x, y) for (x, y) in sparse_waypoints.tolist()]
            dense_waypoints = [QPointF(x, y) for (x, y) in dense_waypoints.tolist()]
            scenario_trigger_points_ = [QPointF(x, y) for (x, y) in scenario_trigger_points.tolist()]
            interpolated_trace = [QPointF(x, y) for (x, y) in interpolated_trace.tolist()]
            other_routes_dense_points = [QPointF(x, y) for (x, y) in other_routes_dense_points.tolist()]

            painter.setPen(
                QPen(
                    self.MAP_COLOR,
                    max(1, self.global_scaling * self.scaling * self.ROAD_WPS_SIZE),
                    Qt.DashDotLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPoints(road_waypoints)

            painter.setPen(
                QPen(
                    self.PARKING_LOT_COLOR,
                    max(1, self.global_scaling * self.scaling * self.ROAD_WPS_SIZE),
                    Qt.DashDotLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPoints(parking_waypoints)

            painter.setPen(
                QPen(
                    self.OTHER_ROUTES_COLOR,
                    max(
                        self.SCALING_OTHER_DENSE_ROUTE,
                        self.global_scaling * self.scaling * self.SCALING_OTHER_DENSE_ROUTE,
                    ),
                    Qt.DashDotLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPoints(other_routes_dense_points)

            painter.setPen(
                QPen(
                    self.ROUTE_COLOR,
                    max(self.SCALING_DENSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_DENSE_ROUTE),
                    Qt.DashDotLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPoints(dense_waypoints)

            painter.setPen(
                QPen(
                    self.NEW_ROUTE_PART_COLOR,
                    max(self.SCALING_NEW_ROUTE_PART, self.global_scaling * self.scaling * self.SCALING_NEW_ROUTE_PART),
                    Qt.DashDotLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPoints(interpolated_trace)

            painter.setPen(
                QPen(
                    self.ROUTE_COLOR,
                    max(self.SCALING_SPARSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_SPARSE_ROUTE),
                    Qt.DashDotLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPoints(sparse_waypoints)

            if sparse_waypoints:
                painter.setPen(
                    QPen(
                        self.FIRST_WP,
                        max(self.SCALING_SPARSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_SPARSE_ROUTE),
                        Qt.DashDotLine,
                        Qt.RoundCap,
                    )
                )
                painter.drawPoints([sparse_waypoints[0]])

            painter.setPen(
                QPen(
                    self.SCENARIO_COLOR,
                    max(
                        self.SCALING_SCENARIO_WAYPOINTS,
                        self.global_scaling * self.scaling * self.SCALING_SCENARIO_WAYPOINTS,
                    ),
                    Qt.DashDotLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPoints(scenario_trigger_points_)

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
                    Qt.DashDotLine,
                    Qt.RoundCap,
                )
            )
            scenario_types = self.selected_route.scenario_types
            p = max(self.SCALING_SPARSE_ROUTE, self.global_scaling * self.scaling * self.SCALING_SPARSE_ROUTE)
            for (x, y), scenario_type in zip(scenario_trigger_points, scenario_types):
                painter.drawText(int(round(x + p)), int(round(y - p)), scenario_type)

            closest_map_coord_screen_coords = [
                QPointF(self.closest_map_coord_screen_coords[0], self.closest_map_coord_screen_coords[1])
            ]
            painter.setPen(
                QPen(
                    self.CURSOR_COLOR,
                    self.global_scaling * self.scaling * self.CURSOR_WP_SIZE,
                    Qt.DashDotLine,
                    Qt.RoundCap,
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

        self.compute_closest_map_coord_in_screen_coords(event.pos())

    def mousePressEvent(self, event):
        """
        Handles mouse press events on the canvas.
        """
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.last_mouse_pos = event.pos()
        elif event.button() == Qt.LeftButton and not self.location_transform_attributes:
            mouse_pos_screen_coords = event.pos()
            mouse_pos_screen_coords = np.array(
                [mouse_pos_screen_coords.x(), mouse_pos_screen_coords.y()], dtype="float"
            )
            mouse_pos_map_coords = self.screen_coords_to_world_coords(mouse_pos_screen_coords[None])[0]

            self.selected_route.add_or_remove_waypoint(mouse_pos_map_coords)
            self.interpolated_trace = []
            self.since_last_mouse_movement = time.time()

            self.parent_obj.update_map_name_and_route_length()
        elif event.button() == Qt.LeftButton:  # Add location or transform for scenario
            self.add_location_data_to_scenario(event.pos())
        elif event.button() == Qt.RightButton and not self.location_transform_attributes:
            mouse_pos_screen_coords = event.pos()
            mouse_pos_screen_coords = np.array(
                [mouse_pos_screen_coords.x(), mouse_pos_screen_coords.y()], dtype="float"
            )
            mouse_pos_map_coords = self.screen_coords_to_world_coords(mouse_pos_screen_coords[None])[0]

            if self.selected_route.check_if_scenario_can_be_added(mouse_pos_map_coords):
                if self.selected_route.should_remove_scenario(mouse_pos_map_coords):
                    self.selected_route.remove_scenario(mouse_pos_map_coords)
                else:
                    scenario_selection_dialog = ScenarioSelectionDialog()
                    selected_scenario = scenario_selection_dialog.selected_scenario

                    if selected_scenario is not None:
                        scenario_attributes = []
                        if config.SCENARIO_TYPES[selected_scenario]:
                            scenario_attribute_dialog = ScenarioAttributeDialog(selected_scenario)
                            scenario_attributes = scenario_attribute_dialog.scenario_attributes

                            self.location_transform_attributes = [
                                x.copy()
                                for x in config.SCENARIO_TYPES[selected_scenario]
                                if "location" in x[1] or "transform" in x[1]
                            ]
                            if self.location_transform_attributes:
                                self.location_transform_attributes.insert(0, selected_scenario)
                            self.prepare_window_to_add_location_data_to_scenario()

                        self.selected_route.add_scenario(mouse_pos_map_coords, selected_scenario, scenario_attributes)
                        self.interpolated_trace = []
                        self.since_last_mouse_movement = time.time()

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
            self.parent_obj.label_add_location.setText(f"Select {first_attribute} for {scenario_type}")
            self.parent_obj.label_add_location.setVisible(True)

    def add_location_data_to_scenario(self, pos):
        """
        Adds location or transform data for a scenario based on the selected point on the canvas.
        """
        idx = 1
        while len(self.location_transform_attributes[idx]) == 3:
            idx += 1

        attr, attr_type = self.location_transform_attributes[idx]
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
        wp_loc = wp.transform.location
        self.location_transform_attributes[idx].append([round(wp_loc.x, 1), round(wp_loc.y, 1), round(wp_loc.z, 1)])

        idx += 1
        if len(self.location_transform_attributes) == idx:
            self.finish_to_add_location_data_to_scenario()
        else:
            scenario_type = self.location_transform_attributes[0]
            first_attribute = self.location_transform_attributes[idx][0]
            self.parent_obj.label_add_location.setText(f"Select {first_attribute} for {scenario_type}")

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
        self.parent_obj.label_add_location.setVisible(False)

        self.selected_route.add_location_transform_attributes_to_last_scenario(self.location_transform_attributes[1:])
        self.location_transform_attributes.clear()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.panning = False

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
        self.compute_closest_map_coord_in_screen_coords(event.pos())

        if self.panning:
            diff = event.pos() - self.last_mouse_pos
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

            self.last_mouse_pos = event.pos()

        # self.update()

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
        # self.update()

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

    # get the closest world coordinate
    def get_closest_road_wp_in_screen_coord(self, mouse_location_screen_coord):
        mouse_location_world_coord = self.screen_coords_to_world_coords(mouse_location_screen_coord[None])[0]
        _, idx = self.tree.query(mouse_location_world_coord)
        closest_map_wp_world_coord = self.all_waypoints_np[idx]
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
        self.setWindowTitle("Route Creator")
        self.resize(800, 640)
        self.center()

        self.carla_client = carla_client
        self.route_manager = RouteManager(carla_client)

        # Creating the main layout and sub-layouts
        main_layout = QHBoxLayout()
        vertical_layout = QVBoxLayout()
        button_layout_empty_save_load = QHBoxLayout()
        button_layout_add_remove = QHBoxLayout()

        # Arranging the layouts
        main_layout.addLayout(vertical_layout)
        vertical_layout.addLayout(button_layout_empty_save_load)
        # Creating and adding buttons to the button layout
        empty_file_button = QPushButton("Empty File")
        empty_file_button.clicked.connect(self.on_empty_file_button_click)
        button_layout_empty_save_load.addWidget(empty_file_button)
        self.empty_file_button = empty_file_button

        load_file_button = QPushButton("Load File")
        load_file_button.clicked.connect(self.on_load_file_button_click)
        button_layout_empty_save_load.addWidget(load_file_button)
        self.load_file_button = load_file_button

        self.save_file_button = QPushButton("Save File")
        self.save_file_button.clicked.connect(self.on_save_file_button_click)
        self.save_file_button.setEnabled(False)
        button_layout_empty_save_load.addWidget(self.save_file_button)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setFixedHeight(10)
        vertical_layout.addWidget(separator)

        self.add_route_button = QPushButton("Add Route")
        self.add_route_button.clicked.connect(self.on_add_route_button_click)
        self.add_route_button.setEnabled(False)
        button_layout_add_remove.addWidget(self.add_route_button)

        self.remove_route_button = QPushButton("Remove Route")
        self.remove_route_button.clicked.connect(self.on_remove_route_button_click)
        self.remove_route_button.setEnabled(False)
        button_layout_add_remove.addWidget(self.remove_route_button)

        self.label_selected_town = QLabel("No town selected")
        self.label_selected_town.setAlignment(Qt.AlignHCenter)
        vertical_layout.addWidget(self.label_selected_town)

        # Adding a list widget to the vertical layout
        self.items_list = QListWidget()
        self.items_list.itemClicked.connect(self.on_list_item_clicked)
        vertical_layout.addWidget(self.items_list)
        vertical_layout.addLayout(button_layout_add_remove)

        font = self.items_list.font()
        font.setPointSize(16)  # Set the font size to 16 points
        self.items_list.setFont(font)

        v_layout2 = QVBoxLayout()
        self.label_add_location = QLabel()
        font2 = self.label_add_location.font()
        self.label_add_location.setStyleSheet("color: red;")
        font2.setPointSize(16)  # Set the font size to 16 points
        self.label_add_location.setFont(font2)

        self.label_add_location.setAlignment(Qt.AlignHCenter)
        v_layout2.addWidget(self.label_add_location)
        self.label_add_location.setVisible(False)

        self.canvas = Canvas(self.carla_client, parent=self, route_manager=self.route_manager)
        self.canvas.setEnabled(False)

        v_layout2.addWidget(self.canvas)
        v_layout2.setStretch(1, 1)  # Allowing the canvas to stretch and take up available space
        main_layout.addLayout(v_layout2)

        # Setting the main layout for the window
        self.setLayout(main_layout)
        main_layout.setStretch(1, 1)  # Allowing the canvas to stretch and take up available space

        # Initialize panning variables
        self.last_mouse_pos = None
        self.panning = False

    def show_yes_no_dialog(self, text):
        reply = QMessageBox.question(self, "Confirmation", text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        return reply == QMessageBox.Yes

    def center(self):
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def on_add_route_button_click(self):
        self.route_manager.add_empty_route()
        self.update_routes_list()
        self.update_map_name_and_route_length()

    def on_remove_route_button_click(self):
        if len(self.route_manager.routes) > 1:
            self.route_manager.remove_selected_route()
            self.update_routes_list()
            self.update_map_name_and_route_length()

    def on_empty_file_button_click(self):
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
                self.save_file_button.setEnabled(True)
                self.remove_route_button.setEnabled(True)
                self.canvas.setEnabled(True)
                self.update_routes_list()
                self.canvas.reset_map_offset_and_scaling()

    def update_routes_list(self):
        routes, selected_route_id = self.route_manager.routes, self.route_manager.selected_route_id
        self.items_list.clear()

        for i, route_id in enumerate(sorted(routes.keys())):
            item = QListWidgetItem(str(route_id))
            item.setTextAlignment(Qt.AlignHCenter)
            self.items_list.addItem(item)

            if route_id == selected_route_id:
                self.items_list.setCurrentRow(i)

        selected_route = self.route_manager.routes[self.route_manager.selected_route_id]
        self.canvas.update_selected_route(selected_route)
        self.canvas.update_data_from_carla_client()

    def closeEvent(self, event):
        if event.spontaneous():
            if self.route_manager.routes:
                close_window = self.show_yes_no_dialog(
                    "Are you sure you want to discard the current routes without saving?"
                )
                if not close_window:
                    event.ignore()
                    return

        event.accept()

    def on_load_file_button_click(self):
        start_load_window = True
        if self.route_manager.routes:
            start_load_window = self.show_yes_no_dialog(
                "Are you sure you want to discard the current routes without saving?"
            )

        if start_load_window:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseCustomDirectoryIcons
            file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "XML Files (*.xml)", options=options)
            if file_name:
                LoadingIndicatorWindow(
                    None, "Loading map...", lambda: self.route_manager.load_routes_from_file(file_name)
                )

                self.update_map_name_and_route_length()
                self.add_route_button.setEnabled(True)
                self.save_file_button.setEnabled(True)
                self.remove_route_button.setEnabled(True)
                self.canvas.setEnabled(True)
                self.update_routes_list()
                self.canvas.reset_map_offset_and_scaling()

    def on_save_file_button_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseCustomDirectoryIcons  # Disable custom icons
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "XML Files (*.xml)", options=options)
        if file_name:
            self.route_manager.save_routes_to_file(file_name)

    def update_map_name_and_route_length(self):
        selected_route = self.route_manager.routes[self.route_manager.selected_route_id]
        self.label_selected_town.setText(
            f"{selected_route.map_name} - {round(selected_route.route_length/1000.,3)} km"
        )

    def on_list_item_clicked(self, item):
        self.route_manager.selected_route_id = int(item.text())  # 3.4809112548828125e-05
        selected_route = self.route_manager.routes[self.route_manager.selected_route_id]  # 4.0531158447265625e-06
        self.canvas.update_selected_route(selected_route)  # 0.00022840499877929688
        # self.canvas.update() # 1.5974044799804688e-05

        self.update_map_name_and_route_length()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="The IP for the CARLA simulator")
    parser.add_argument("--port", type=int, default=2000, help="The IP for the CARLA simulator")
    parser.add_argument(
        "--map-data-dir", type=str, default="carla_map_data", help="The path of the directory with the map data"
    )
    args = parser.parse_args()

    carla_client = CarlaClient(args.host, args.port, args.map_data_dir)

    app = QApplication(sys.argv)
    main_window = Window(carla_client)
    main_window.show()
    sys.exit(app.exec_())
