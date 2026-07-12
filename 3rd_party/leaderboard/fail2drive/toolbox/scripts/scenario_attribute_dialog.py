"""
This module provides a PyQt6 dialog for configuring scenario attributes
for CARLA. The dialog allows users to input various types of data, such as integer values, intervals,
and choices, depending on the scenario type. The user's input is stored
in a list of tuples, where each tuple contains the attribute name, attribute type, and the corresponding input value(s).
"""

import sys
from PyQt6.QtWidgets import (
    QComboBox,
    QLineEdit,
    QLabel,
    QDialog,
    QApplication,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import config


class DefaultValueLineEdit(QLineEdit):
    """
    A QLineEdit that can show an auto-default text and replace it on first user edit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_default_active = False
        self._default_text = None
        self.textEdited.connect(self._on_text_edited)
        self.editingFinished.connect(self._restore_default_if_empty)

    def set_auto_default(self, value):
        if value is None:
            return
        self._default_text = str(value)
        self.setText(self._default_text)
        self._auto_default_active = True

    def _on_text_edited(self, _):
        self._auto_default_active = False

    def _restore_default_if_empty(self):
        if self._default_text is None:
            return
        if not self.text().strip():
            self.setText(self._default_text)
            self._auto_default_active = True

    def effective_text(self):
        text = self.text()
        if text.strip():
            return text
        if self._default_text is not None:
            return self._default_text
        return text

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._auto_default_active:
            QTimer.singleShot(0, self.selectAll)

    def focusOutEvent(self, event):
        self._restore_default_if_empty()
        super().focusOutEvent(event)


class ScenarioAttributeDialog(QDialog):
    def __init__(self, scenario_type, parent=None, font_size=14, initial_values=None, allow_layout_customization=False):
        super().__init__(parent)
        self.setWindowTitle(scenario_type)

        self.scenario_attributes = []  # List to store scenario attributes
        self.was_cancelled = True
        self.default_values = {}

        self.scenario_type = scenario_type
        self.font_size = font_size
        self.initial_values = initial_values or {}
        self.allow_layout_customization = bool(allow_layout_customization)
        self.request_layout_customization = False
        self.requested_map_pick = None
        self.requested_map_pick_initial_values = None

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        if self.scenario_type == "BackgroundActivityParametrizer":
            note_label = QLabel("Note: Leave empty to reset.")
            note_font = QFont("Arial", max(10, self.font_size - 1))
            note_label.setFont(note_font)
            note_label.setWordWrap(True)
            self.main_layout.addWidget(note_label)

        self.left_layout = QVBoxLayout()  # Layout for labels
        self.right_layout = QVBoxLayout()  # Layout for input widgets

        # Add left and right layouts to a horizontal layout
        vlayout = QHBoxLayout()
        self.main_layout.addLayout(vlayout)
        vlayout.addLayout(self.left_layout)
        vlayout.addLayout(self.right_layout)

        # Add input widgets for each scenario attribute
        hidden_layout_managed_attrs = set()
        if scenario_type in ("PermutedConstructionObstacle", "PermutedConstructionObstacleTwoWays"):
            hidden_layout_managed_attrs = {"warning_sign", "debris", "cones"}
        elif scenario_type in ("BadParkingObstacle", "BadParkingObstacleTwoWays"):
            hidden_layout_managed_attrs = {"vehicle", "x", "y", "yaw"}
        for field_def in config.SCENARIO_TYPES[scenario_type]:
            attribute = field_def[0]
            if attribute in hidden_layout_managed_attrs:
                continue
            attr_type = field_def[1]
            default_value = None if len(field_def) == 2 else field_def[2]
            self.default_values[attribute] = default_value
            tooltip = config.get_scenario_param_tooltip(self.scenario_type, attribute)
            self.add_input_widget(attribute, attr_type, default_value, tooltip)

        # Add select button
        button_layout = QHBoxLayout()
        if self.allow_layout_customization:
            customize_button = QPushButton("Customize Layout")
            customize_button.clicked.connect(self.customize_layout)
            button_layout.addWidget(customize_button)
        select_button = QPushButton("Save")
        select_button.clicked.connect(self.set_attributes_before_closing)
        button_layout.addWidget(select_button)
        self.main_layout.addLayout(button_layout)

        # Resize and center the dialog
        self.resize(self.sizeHint().width(), self.sizeHint().height())
        self.center()

        self.setModal(True)

    def set_attributes_before_closing(self):
        """
        Set the scenario attributes based on the user input
        before closing the dialog.
        """
        missing_map_fields = []
        updated_attributes = []
        for attr_entry in self.scenario_attributes:
            if not attr_entry:
                continue
            attribute, attr_type, input_widgets = attr_entry

            if attr_type in ("bool", "value"):
                line_edit = input_widgets
                try:
                    text = line_edit.effective_text().strip() if isinstance(line_edit, DefaultValueLineEdit) else line_edit.text().strip()
                    default_value = self.default_values.get(attribute)
                    if not text:
                        if default_value is None:
                            updated_attributes.append(None)
                            continue
                        value = self._coerce_scalar(default_value)
                    else:
                        value = self._parse_scalar(text)
                    updated_attributes.append((attribute, attr_type, value))
                except ValueError:
                    updated_attributes.append(None)

            elif attr_type == "transform":
                picker_state = input_widgets if isinstance(input_widgets, dict) else None
                if picker_state and picker_state.get("value") is not None:
                    updated_attributes.append((attribute, attr_type, picker_state["value"]))
                else:
                    missing_map_fields.append(attribute)
                    updated_attributes.append(None)

            elif "location" in attr_type:
                picker_state = input_widgets if isinstance(input_widgets, dict) else None
                if picker_state and picker_state.get("value") is not None:
                    updated_attributes.append((attribute, attr_type, picker_state["value"]))
                else:
                    missing_map_fields.append(attribute)
                    updated_attributes.append(None)

            elif attr_type == "interval":
                try:
                    line_edit_from, line_edit_to = input_widgets
                    from_text = (
                        line_edit_from.effective_text().strip()
                        if isinstance(line_edit_from, DefaultValueLineEdit)
                        else line_edit_from.text().strip()
                    )
                    to_text = (
                        line_edit_to.effective_text().strip()
                        if isinstance(line_edit_to, DefaultValueLineEdit)
                        else line_edit_to.text().strip()
                    )
                    default_value = self.default_values.get(attribute)
                    default_from = default_value[0] if isinstance(default_value, (list, tuple)) and len(default_value) >= 2 else None
                    default_to = default_value[1] if isinstance(default_value, (list, tuple)) and len(default_value) >= 2 else None

                    if not from_text and default_from is None:
                        updated_attributes.append(None)
                        continue
                    if not to_text and default_to is None:
                        updated_attributes.append(None)
                        continue

                    from_value = self._parse_scalar(from_text) if from_text else self._coerce_scalar(default_from)
                    to_value = self._parse_scalar(to_text) if to_text else self._coerce_scalar(default_to)
                    updated_attributes.append((attribute, attr_type, [from_value, to_value]))
                except ValueError:
                    updated_attributes.append(None)

            elif attr_type == "choice":
                try:
                    combo_box = input_widgets
                    direction = combo_box.currentText()
                    updated_attributes.append((attribute, attr_type, direction))
                except:
                    updated_attributes.append(None)

            else:
                raise NotImplementedError(f"Type {attr_type} is not implemented yet")

        if missing_map_fields:
            field_list = ", ".join(sorted(set(missing_map_fields)))
            QMessageBox.warning(
                self,
                "Missing Waypoints",
                f"Select map waypoints for: {field_list}",
            )
            return

        # Remove None values from the scenario_attributes list
        self.scenario_attributes = [attr for attr in updated_attributes if attr is not None]
        self.was_cancelled = False
        self.accept()

    def customize_layout(self):
        self.request_layout_customization = True
        self.set_attributes_before_closing()

    def center(self):
        """
        Center the dialog on the screen.
        """
        frame_geometry = self.frameGeometry()
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            center_point = screen.availableGeometry().center()
            frame_geometry.moveCenter(center_point)
            self.move(frame_geometry.topLeft())

    def add_input_widget(self, attribute, attr_type, default_value, tooltip):
        """
        Add input widgets for the given attribute and type.
        """
        if attr_type in ("bool", "value"):

            label = QLabel(f"{attribute.upper()}: ")
            label.setToolTip(tooltip)
            self.left_layout.addWidget(label)

            font = QFont("Arial", self.font_size)
            label.setFont(font)

            line_edit = DefaultValueLineEdit()
            line_edit.setPlaceholderText("" if default_value is None else str(default_value))
            line_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            line_edit.setToolTip(tooltip)
            if attribute in self.initial_values:
                line_edit.setText(str(self.initial_values[attribute]))
            elif default_value is not None:
                line_edit.set_auto_default(default_value)
            self.right_layout.addWidget(line_edit)
            line_edit.setFont(font)

            self.scenario_attributes.append((attribute, attr_type, line_edit))

        elif attr_type == "transform":
            font = QFont("Arial", self.font_size)
            label = QLabel(f"{attribute.upper()}: ")
            label.setToolTip(tooltip)
            label.setFont(font)
            self.left_layout.addWidget(label)

            right_row = QHBoxLayout()
            self.right_layout.addLayout(right_row)

            value_label = QLabel(self._format_location_value(self.initial_values.get(attribute)))
            value_label.setToolTip(tooltip)
            value_label.setFont(font)
            right_row.addWidget(value_label)

            pick_button = QPushButton("Select on Map")
            pick_button.setToolTip(tooltip)
            pick_button.clicked.connect(lambda _, a=attribute, t=attr_type: self.request_map_pick_for_attribute(a, t))
            right_row.addWidget(pick_button)

            current_value = self.initial_values.get(attribute)
            if not (isinstance(current_value, (list, tuple)) and len(current_value) >= 3):
                current_value = None
            self.scenario_attributes.append(
                (
                    attribute,
                    attr_type,
                    {
                        "value": list(current_value) if current_value is not None else None,
                        "value_label": value_label,
                    },
                )
            )

        elif "location" in attr_type:
            font = QFont("Arial", self.font_size)
            label = QLabel(f"{attribute.upper()}: ")
            label.setToolTip(tooltip)
            label.setFont(font)
            self.left_layout.addWidget(label)

            right_row = QHBoxLayout()
            self.right_layout.addLayout(right_row)

            value_label = QLabel(self._format_location_value(self.initial_values.get(attribute)))
            value_label.setToolTip(tooltip)
            value_label.setFont(font)
            right_row.addWidget(value_label)

            pick_button = QPushButton("Select on Map")
            pick_button.setToolTip(tooltip)
            pick_button.clicked.connect(lambda _, a=attribute, t=attr_type: self.request_map_pick_for_attribute(a, t))
            right_row.addWidget(pick_button)

            current_value = self.initial_values.get(attribute)
            if not (isinstance(current_value, (list, tuple)) and len(current_value) >= 3):
                current_value = None
            self.scenario_attributes.append(
                (
                    attribute,
                    attr_type,
                    {
                        "value": list(current_value) if current_value is not None else None,
                        "value_label": value_label,
                    },
                )
            )

        elif attr_type == "interval":
            font = QFont("Arial", self.font_size)

            label = QLabel(f"{attribute.upper()}: ")
            label.setToolTip(tooltip)
            self.left_layout.addWidget(label)
            label.setFont(font)

            h_layout = QHBoxLayout()
            self.right_layout.addLayout(h_layout)

            line_edit_from = DefaultValueLineEdit()
            from_placeholder = ""
            to_placeholder = ""
            if isinstance(default_value, (list, tuple)) and len(default_value) >= 2:
                from_placeholder = str(default_value[0])
                to_placeholder = str(default_value[1])
            line_edit_from.setPlaceholderText(from_placeholder)
            line_edit_from.setFont(font)
            line_edit_from.setAlignment(Qt.AlignmentFlag.AlignCenter)
            line_edit_from.setToolTip(tooltip)
            if attribute in self.initial_values and len(self.initial_values[attribute]) >= 2:
                line_edit_from.setText(str(self.initial_values[attribute][0]))
            elif isinstance(default_value, (list, tuple)) and len(default_value) >= 2:
                line_edit_from.set_auto_default(default_value[0])
            h_layout.addWidget(line_edit_from)

            h_layout.addWidget(QLabel("-"))

            line_edit_to = DefaultValueLineEdit()
            line_edit_to.setPlaceholderText(to_placeholder)
            line_edit_to.setFont(font)
            line_edit_to.setAlignment(Qt.AlignmentFlag.AlignCenter)
            line_edit_to.setToolTip(tooltip)
            if attribute in self.initial_values and len(self.initial_values[attribute]) >= 2:
                line_edit_to.setText(str(self.initial_values[attribute][1]))
            elif isinstance(default_value, (list, tuple)) and len(default_value) >= 2:
                line_edit_to.set_auto_default(default_value[1])
            h_layout.addWidget(line_edit_to)

            self.scenario_attributes.append((attribute, attr_type, (line_edit_from, line_edit_to)))

        elif attr_type == "choice":
            font = QFont("Arial", self.font_size)

            label = QLabel(f"{attribute.upper()}: ")
            label.setToolTip(tooltip)
            self.left_layout.addWidget(label)
            label.setFont(font)

            combo_box = QComboBox(self)
            combo_box.setFont(font)
            combo_box.setToolTip(tooltip)
            combo_box.addItem("right")
            combo_box.addItem("left")
            if attribute in self.initial_values and self.initial_values[attribute] in ("left", "right"):
                combo_box.setCurrentText(self.initial_values[attribute])
            self.right_layout.addWidget(combo_box)

            self.scenario_attributes.append((attribute, attr_type, combo_box))

        else:
            raise NotImplementedError(f"Type {attr_type} is not implemented yet")

    @staticmethod
    def _parse_scalar(text):
        value = str(text).strip()
        if not value:
            raise ValueError("empty")
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

    @staticmethod
    def _coerce_scalar(value):
        if isinstance(value, str):
            return ScenarioAttributeDialog._parse_scalar(value)
        return value

    @staticmethod
    def _format_location_value(value):
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            xyz = [round(float(value[0]), 1), round(float(value[1]), 1), round(float(value[2]), 1)]
            return f"x={xyz[0]}, y={xyz[1]}, z={xyz[2]}"
        return "Not selected"

    def request_map_pick_for_attribute(self, attribute, attr_type):
        self.requested_map_pick = (attribute, attr_type)
        self.requested_map_pick_initial_values = self._collect_current_values()
        self.was_cancelled = False
        self.accept()

    def _collect_current_values(self):
        values = {}
        for attr_entry in self.scenario_attributes:
            if not attr_entry:
                continue
            attribute, attr_type, input_widgets = attr_entry
            if attr_type in ("bool", "value"):
                line_edit = input_widgets
                text = line_edit.text().strip()
                if text:
                    values[attribute] = text
            elif attr_type == "interval":
                line_edit_from, line_edit_to = input_widgets
                from_text = line_edit_from.text().strip()
                to_text = line_edit_to.text().strip()
                if from_text or to_text:
                    values[attribute] = [from_text, to_text]
            elif attr_type == "choice":
                values[attribute] = input_widgets.currentText()
            elif attr_type == "transform" or "location" in attr_type:
                if isinstance(input_widgets, dict) and input_widgets.get("value") is not None:
                    values[attribute] = list(input_widgets["value"])
        return values


if __name__ == "__main__":
    app = QApplication(sys.argv)
    scenario_type = "SignalizedJunctionLeftTurn"
    scenario_attribute_dialog = ScenarioAttributeDialog(scenario_type)
    scenario_attribute_dialog.exec()
    if not scenario_attribute_dialog.was_cancelled:
        print(scenario_attribute_dialog.scenario_attributes)
    sys.exit(app.exec())
