from PyQt6.QtCore import Qt, QEvent, QSignalBlocker
from PyQt6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pathlib import Path
import math
import sys


_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from customobstacle_designer import Designer, ObstacleItem  # noqa: E402


class PermutedConstructionLayoutDialog(QDialog):
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

        self.warning_combo: QComboBox | None = None
        self.debris_combo: QComboBox | None = None
        self.cone_toggles = []
        self._last_warning_non_none = str(self.initial_values.get("warning_sign", "static.prop.trafficwarning"))
        self._last_debris_non_none = str(self.initial_values.get("debris", "static.prop.dirtdebris02"))
        self._toggle_targets_cache = {}

        self.designer = Designer()
        self.designer.configure_editor_profile({"object_mode": "construction", "overlay_types": ["Construction"]})
        self.designer.configure_for_altered_construction_editor()
        # In permuted-construction mode, clicks are used for toggle/remove behavior.
        # Disable double-click prop edit flow to avoid interaction conflicts.
        self.designer.view.on_obstacle_double_click = None
        self.warning_combo = self.designer.make_static_prop_selector(
            str(self.initial_values.get("warning_sign", "static.prop.trafficwarning"))
        )
        self.debris_combo = self.designer.make_static_prop_selector(
            str(self.initial_values.get("debris", "static.prop.dirtdebris02"))
        )
        self.warning_combo.insertItem(0, "none")
        self.debris_combo.insertItem(0, "none")
        cones_widget = QWidget()
        cones_layout = QHBoxLayout(cones_widget)
        cones_layout.setContentsMargins(0, 0, 0, 0)
        initial_cones = self._normalize_cones_bitstring(self.initial_values.get("cones", "1111111"))
        for i in range(7):
            toggle = QCheckBox(str(i + 1))
            is_on = i < len(initial_cones) and initial_cones[i] == "1"
            toggle.setChecked(is_on)
            toggle.toggled.connect(lambda _checked: self._update_preview())
            self.cone_toggles.append(toggle)
            cones_layout.addWidget(toggle)
        cones_layout.addStretch(1)
        params_group = QGroupBox("Permuted Construction Params")
        params_layout = QFormLayout(params_group)
        params_layout.addRow(QLabel("WARNING_SIGN"), self.warning_combo)
        params_layout.addRow(QLabel("DEBRIS"), self.debris_combo)
        params_layout.addRow(QLabel("CONES"), cones_widget)
        self.designer.set_parameter_widgets([("", params_group)])
        self.designer.distance_row.setVisible(True)
        self.designer.set_overlay_selection("construction", str(self.initial_values.get("direction", "right")))
        self.designer.set_construction_preview_params(
            warning_sign=self.warning_combo.currentText().strip(),
            debris=self.debris_combo.currentText().strip(),
            cones=self._cones_bitstring(),
        )
        self.warning_combo.currentTextChanged.connect(lambda _x: self._update_preview())
        self.debris_combo.currentTextChanged.connect(lambda _x: self._update_preview())
        self.designer.overlay_direction.currentTextChanged.connect(self._on_direction_changed)
        self.designer.view.viewport().installEventFilter(self)
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

    def _update_preview(self):
        warning_val = self.warning_combo.currentText().strip()
        debris_val = self.debris_combo.currentText().strip()
        if warning_val and warning_val != "none":
            self._last_warning_non_none = warning_val
        if debris_val and debris_val != "none":
            self._last_debris_non_none = debris_val
        self.designer.set_construction_preview_params(
            warning_sign=warning_val,
            debris=debris_val,
            cones=self._cones_bitstring(),
        )

    def _on_direction_changed(self, _value):
        self._toggle_targets_cache.clear()
        self._update_preview()

    def set_attributes_before_closing(self):
        updated = {name: [attr, atype, val] for (attr, atype, val) in self.base_attributes for name in [attr]}
        updated["warning_sign"] = ["warning_sign", "value", self.warning_combo.currentText().strip() or "none"]
        updated["debris"] = ["debris", "value", self.debris_combo.currentText().strip() or "none"]
        updated["cones"] = ["cones", "value", self._cones_bitstring()]

        direction = self.designer.overlay_direction.currentText().strip()
        if "direction" in updated:
            updated["direction"] = ["direction", "choice", "left" if direction == "left" else "right"]

        self.scenario_attributes = list(updated.values())
        self.was_cancelled = False
        self.accept()

    def _cones_bitstring(self):
        if not self.cone_toggles:
            return "1111111"
        return "".join("1" if toggle.isChecked() else "0" for toggle in self.cone_toggles)

    @staticmethod
    def _normalize_cones_bitstring(value):
        raw = "".join(ch for ch in str(value) if ch in ("0", "1"))
        if not raw:
            return "1111111"
        if len(raw) >= 7:
            return raw[:7]
        return raw.ljust(7, "0")

    def _on_scene_selection_remove(self):
        selected = [i for i in self.designer.scene.selectedItems() if isinstance(i, ObstacleItem)]
        if not selected:
            return
        clicked = selected[0].obstacle
        object_id = str(clicked.object_id).lower()
        changed = False

        if "trafficwarning" in object_id or "warningconstruction" in object_id:
            self.warning_combo.setCurrentText("none")
            changed = True
        elif "dirtdebris" in object_id:
            self.debris_combo.setCurrentText("none")
            changed = True
        elif "cone" in object_id:
            cone_idx = self._closest_cone_index(clicked.x, clicked.y)
            if cone_idx is not None and 0 <= cone_idx < len(self.cone_toggles):
                self.cone_toggles[cone_idx].setChecked(False)
                changed = True

        self.designer.scene.clearSelection()
        self.designer.list_widget.clearSelection()
        if changed:
            self._update_preview()

    def _closest_cone_index(self, x, y):
        full = self.designer._overlay_construction(0.0, 3.5, self.designer.overlay_direction.currentText(), preview_params={
            "warning_sign": "none",
            "debris": "none",
            "cones": "1111111",
        })
        cones = [obj for obj in full if "cone" in str(obj.get("object_id", "")).lower()]
        if not cones:
            return None
        best_i = None
        best_d = None
        for i, cone in enumerate(cones):
            dx = float(cone["x"]) - float(x)
            dy = float(cone["y"]) - float(y)
            d2 = dx * dx + dy * dy
            if best_d is None or d2 < best_d:
                best_d = d2
                best_i = i
        return best_i

    def eventFilter(self, watched, event):
        if watched is self.designer.view.viewport() and event.type() == QEvent.Type.MouseButtonDblClick:
            # Ignore double-click semantic path; toggling is handled on release for each click.
            return True

        if watched is self.designer.view.viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() != Qt.MouseButton.LeftButton:
                return super().eventFilter(watched, event)

            scene_pos = self.designer.view.mapToScene(event.pos())
            x = float(-scene_pos.y() / self.designer.scale)
            y = float(scene_pos.x() / self.designer.scale)
            action = self._closest_toggle_target(x, y)
            if action is None:
                return super().eventFilter(watched, event)

            kind, idx = action
            if kind == "warning":
                with QSignalBlocker(self.warning_combo):
                    if self.warning_combo.currentText().strip() == "none":
                        self.warning_combo.setCurrentText(self._last_warning_non_none or "static.prop.trafficwarning")
                    else:
                        self.warning_combo.setCurrentText("none")
            elif kind == "debris":
                with QSignalBlocker(self.debris_combo):
                    if self.debris_combo.currentText().strip() == "none":
                        self.debris_combo.setCurrentText(self._last_debris_non_none or "static.prop.dirtdebris02")
                    else:
                        self.debris_combo.setCurrentText("none")
            elif kind == "cone" and idx is not None and 0 <= idx < len(self.cone_toggles):
                with QSignalBlocker(self.cone_toggles[idx]):
                    self.cone_toggles[idx].setChecked(not self.cone_toggles[idx].isChecked())

            self._update_preview()
            self.designer.scene.clearSelection()
            self.designer.list_widget.clearSelection()
            return True

        return super().eventFilter(watched, event)

    def _closest_toggle_target(self, x, y):
        direction = self.designer.overlay_direction.currentText()
        warning_for_hit = self.warning_combo.currentText().strip()
        debris_for_hit = self.debris_combo.currentText().strip()
        if not warning_for_hit or warning_for_hit == "none":
            warning_for_hit = self._last_warning_non_none or "static.prop.trafficwarning"
        if not debris_for_hit or debris_for_hit == "none":
            debris_for_hit = self._last_debris_non_none or "static.prop.dirtdebris02"

        cache_key = (direction, warning_for_hit, debris_for_hit)
        full = self._toggle_targets_cache.get(cache_key)
        if full is None:
            full = self.designer._overlay_construction(
                0.0,
                3.5,
                direction,
                preview_params={
                    "warning_sign": warning_for_hit,
                    "debris": debris_for_hit,
                    "cones": "1111111",
                },
            )
            self._toggle_targets_cache[cache_key] = full
        candidates = []
        cone_idx = 0
        for obj in full:
            kind = str(obj.get("kind", "")).lower()
            cx = float(obj["x"])
            cy = float(obj["y"])
            yaw = float(obj.get("yaw", 0.0))
            ex = float(obj.get("extent_x", 0.25))
            ey = float(obj.get("extent_y", 0.25))
            if kind == "warning":
                candidates.append(("warning", None, cx, cy, yaw, ex, ey))
            elif kind == "debris":
                candidates.append(("debris", None, cx, cy, yaw, ex, ey))
            elif kind == "cone":
                candidates.append(("cone", cone_idx, cx, cy, yaw, ex, ey))
                cone_idx += 1

        hit_padding = 0.12
        best = None
        best_d = None
        for kind, idx, cx, cy, yaw, ex, ey in candidates:
            if not self._point_in_oriented_box(x, y, cx, cy, yaw, ex + hit_padding, ey + hit_padding):
                continue
            dx = cx - x
            dy = cy - y
            d2 = dx * dx + dy * dy
            if best_d is None or d2 < best_d:
                best_d = d2
                best = (kind, idx)
        return best

    @staticmethod
    def _point_in_oriented_box(px, py, cx, cy, yaw_deg, extent_x, extent_y):
        # Transform point into local object frame (CARLA x-forward, y-right).
        dx = float(px) - float(cx)
        dy = float(py) - float(cy)
        rad = math.radians(float(yaw_deg))
        c = math.cos(rad)
        s = math.sin(rad)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        return abs(local_x) <= float(extent_x) and abs(local_y) <= float(extent_y)
