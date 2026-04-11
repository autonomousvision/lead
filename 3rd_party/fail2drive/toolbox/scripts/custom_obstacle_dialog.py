import json
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from customobstacle_designer import Designer  # noqa: E402


class CustomObstacleDialog(QDialog):
    PRESETS_PATH = Path(".autosave") / "custom_obstacle_presets.json"

    def __init__(
        self,
        scenario_type,
        parent=None,
        font_size=14,
        initial_values=None,
        editor_profile=None,
        base_attributes=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(scenario_type)
        self.resize(1300, 820)

        self.scenario_type = scenario_type
        self.font_size = font_size
        self.initial_values = initial_values or {}
        self.base_attributes = [tuple(x) for x in (base_attributes or [])]
        self.scenario_attributes = []
        self.was_cancelled = True
        self.presets = {}
        self.editor_profile = editor_profile or {}

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self._build_preset_controls(controls_layout)

        main_layout.addWidget(controls_widget)

        self.designer = Designer()
        self.designer.configure_for_route_builder()
        self.designer.configure_editor_profile(self.editor_profile)
        self.designer.set_parameter_widgets([])
        self.designer.distance_row.setVisible(False)
        main_layout.addWidget(self.designer, stretch=1)

        button_row = QHBoxLayout()
        select_button = QPushButton("Save")
        select_button.clicked.connect(self.set_attributes_before_closing)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(select_button)
        button_row.addWidget(cancel_button)
        main_layout.addLayout(button_row)

        self._load_presets()
        self._apply_initial_values()

        self.setModal(True)

    def _build_preset_controls(self, parent_layout):
        preset_row = QHBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(220)
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("Preset name")

        load_button = QPushButton("Load Preset")
        load_button.clicked.connect(self.on_load_preset)
        save_button = QPushButton("Save Preset")
        save_button.clicked.connect(self.on_save_preset)
        delete_button = QPushButton("Delete Preset")
        delete_button.clicked.connect(self.on_delete_preset)

        preset_row.addWidget(QLabel("Presets"))
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(self.preset_name_edit)
        preset_row.addWidget(load_button)
        preset_row.addWidget(save_button)
        preset_row.addWidget(delete_button)
        parent_layout.addLayout(preset_row)

    def _apply_initial_values(self):
        overlay = str(self.initial_values.get("baseline_overlay", "construction")).lower()
        if overlay == "none":
            overlay = "construction"
        direction = str(self.initial_values.get("overlay_direction", "right")).lower()
        self.designer.set_overlay_selection(overlay, direction)

        objects_map = self.initial_values.get("objects")
        if isinstance(objects_map, dict):
            self.designer.load_from_objects_attr_map(objects_map)

    def _load_presets(self):
        self.PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if self.PRESETS_PATH.exists():
            try:
                data = json.loads(self.PRESETS_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.presets = data
            except (OSError, json.JSONDecodeError):
                self.presets = {}

        self.preset_combo.clear()
        self.preset_combo.addItems(sorted(self.presets.keys()))

    def _save_presets(self):
        self.PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.PRESETS_PATH.write_text(json.dumps(self.presets, indent=2), encoding="utf-8")

    def _collect_current_state(self):
        return {
            "baseline_overlay": self.designer.get_overlay_selection()[0],
            "overlay_direction": self.designer.get_overlay_selection()[1],
            "objects": self.designer.get_objects_attr_map(),
        }

    def on_load_preset(self):
        name = self.preset_combo.currentText().strip()
        if not name or name not in self.presets:
            return

        preset = self.presets[name]
        overlay = str(preset.get("baseline_overlay", "construction")).lower()
        if overlay == "none":
            overlay = "construction"
        direction = str(preset.get("overlay_direction", "right")).lower()
        self.designer.set_overlay_selection(overlay, direction)

        objects = preset.get("objects", {})
        if isinstance(objects, dict):
            self.designer.load_from_objects_attr_map(objects)

    def on_save_preset(self):
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a preset name.")
            return

        self.presets[name] = self._collect_current_state()
        self._save_presets()
        self._load_presets()
        self.preset_combo.setCurrentText(name)
        self.preset_name_edit.clear()

    def on_delete_preset(self):
        name = self.preset_combo.currentText().strip()
        if not name or name not in self.presets:
            return

        del self.presets[name]
        self._save_presets()
        self._load_presets()

    def set_attributes_before_closing(self):
        self.scenario_attributes = [tuple(x) for x in self.base_attributes]

        baseline_overlay, overlay_direction = self.designer.get_overlay_selection()
        objects_map = self.designer.get_objects_attr_map()

        self.scenario_attributes.append(("baseline_overlay", "choice", baseline_overlay))
        self.scenario_attributes.append(("overlay_direction", "choice", overlay_direction))
        self.scenario_attributes.append(("objects", "objects", objects_map))

        self.was_cancelled = False
        self.accept()
