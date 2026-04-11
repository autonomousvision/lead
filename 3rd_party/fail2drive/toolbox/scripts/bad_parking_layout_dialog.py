from pathlib import Path
import sys

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from customobstacle_designer import Designer, DEFAULT_LANE_WIDTH, to_scene_xy  # noqa: E402


class BadParkingLayoutDialog(QDialog):
    def __init__(self, scenario_type, parent=None, initial_values=None, base_attributes=None):
        super().__init__(parent)
        self.setWindowTitle(scenario_type)
        self.resize(1300, 820)

        self.initial_values = initial_values or {}
        self.base_attributes = [tuple(x) for x in (base_attributes or [])]
        self.scenario_attributes = []
        self.was_cancelled = True

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.designer = Designer()
        self.designer.configure_editor_profile({"object_mode": "vehicle_only", "overlay_types": ["ParkedVehicle"]})
        self.designer.configure_for_bad_parking_editor()
        self.designer.view.on_delete = None

        self.vehicle_combo = QComboBox()
        self.vehicle_combo.setMinimumContentsLength(30)
        self.vehicle_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.vehicle_combo.addItems(self.designer.available_object_ids())
        self.vehicle_combo.currentTextChanged.connect(self._on_vehicle_changed)

        params_group = QGroupBox("Bad Parking Params")
        params_layout = QFormLayout(params_group)
        params_layout.addRow(QLabel("VEHICLE"), self.vehicle_combo)
        self.designer.set_parameter_widgets([("", params_group)])

        initial_direction = str(self.initial_values.get("direction", "right")).strip().lower()
        if initial_direction not in ("left", "right"):
            initial_direction = "right"
        self.designer.set_overlay_selection("parkedvehicle", initial_direction)
        self.designer.overlay_enabled.setChecked(True)
        self.designer.overlay_enabled.setEnabled(False)

        self.has_direction_attr = any(attr == "direction" for attr, _atype, _value in self.base_attributes)
        self.designer.overlay_direction.setEnabled(self.has_direction_attr)

        self._seed_single_vehicle()
        self.designer.overlay_direction.currentTextChanged.connect(self._on_direction_changed)
        main_layout.addWidget(self.designer, stretch=1)

        button_row = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.set_attributes_before_closing)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(save_button)
        button_row.addWidget(cancel_button)
        main_layout.addLayout(button_row)

        self.setModal(True)

    def _safe_float(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _direction_sign(self):
        direction = self.designer.overlay_direction.currentText().strip().lower()
        return -1.0 if direction == "left" else 1.0

    def _resolve_vehicle_id(self, raw_vehicle_id: str):
        raw = str(raw_vehicle_id or "").strip()
        available = self.designer.available_object_ids()
        if not available:
            return ""
        if raw in available:
            return raw
        if "*" in raw:
            prefix = raw.split("*", 1)[0]
            for object_id in available:
                if object_id.startswith(prefix):
                    return object_id
        for fallback in ("vehicle.audi.tt", "vehicle.tesla.model3", "vehicle.nissan.patrol_2021"):
            if fallback in available:
                return fallback
        return available[0]

    def _seed_single_vehicle(self):
        object_id = self._resolve_vehicle_id(self.initial_values.get("vehicle", "vehicle.*"))
        x = self._safe_float(self.initial_values.get("x", 0.0), 0.0)
        default_y = 0.7 * DEFAULT_LANE_WIDTH / 2.0
        if "y" in self.initial_values:
            y = self._safe_float(self.initial_values.get("y", default_y), default_y)
        else:
            y = default_y * self._direction_sign()
        yaw = self._safe_float(self.initial_values.get("yaw", 0.0), 0.0)

        self.designer.clear_all_objects()
        self.designer.add_obstacle_with_params(object_id, x, y, yaw, allow_disallowed=True)
        if self.designer.list_widget.count() > 0:
            self.designer.list_widget.setCurrentRow(0)
            self.designer.sync_selection_from_list()
        self._align_obstacle_side()
        self.vehicle_combo.blockSignals(True)
        if self.vehicle_combo.findText(object_id) >= 0:
            self.vehicle_combo.setCurrentText(object_id)
        self.vehicle_combo.blockSignals(False)

    def _current_obstacle(self):
        if self.designer.obstacles:
            return self.designer.obstacles[0]
        return None

    def _on_vehicle_changed(self, object_id: str):
        obs = self._current_obstacle()
        if obs is None:
            return
        self.designer.set_obstacle_object_id(obs, str(object_id).strip())

    def _align_obstacle_side(self):
        obs = self._current_obstacle()
        if obs is None:
            return
        side_sign = self._direction_sign()
        y_aligned = abs(float(obs.y)) * side_sign
        if float(obs.y) == y_aligned:
            return
        obs.y = y_aligned
        item = self.designer.items.get(obs.label)
        if item is not None:
            item.setPos(to_scene_xy(obs.x, obs.y, self.designer.scale))
        self.designer.sync_selection_from_list()

    def _on_direction_changed(self, _value):
        self._align_obstacle_side()

    def set_attributes_before_closing(self):
        obs = self._current_obstacle()
        if obs is None:
            return

        updated = {name: [attr, atype, val] for (attr, atype, val) in self.base_attributes for name in [attr]}
        updated["vehicle"] = ["vehicle", "value", str(obs.object_id)]
        updated["x"] = ["x", "value", round(float(obs.x), 2)]
        y_aligned = abs(float(obs.y)) * self._direction_sign()
        updated["y"] = ["y", "value", round(y_aligned, 2)]
        updated["yaw"] = ["yaw", "value", round(float(obs.yaw), 2)]

        if "direction" in updated:
            direction = self.designer.overlay_direction.currentText().strip().lower()
            updated["direction"] = ["direction", "choice", "left" if direction == "left" else "right"]

        self.scenario_attributes = list(updated.values())
        self.was_cancelled = False
        self.accept()
