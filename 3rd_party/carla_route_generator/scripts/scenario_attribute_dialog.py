"""
This module provides a PyQt5 dialog for configuring scenario attributes
for CARLA. The dialog allows users to input various types of data, such as integer values, intervals,
and choices, depending on the scenario type. The user's input is stored
in a list of tuples, where each tuple contains the attribute name, attribute type, and the corresponding input value(s).
"""

import sys
from PyQt5.QtWidgets import (
    QComboBox,
    QLineEdit,
    QLabel,
    QDialog,
    QApplication,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIntValidator
import config


class ScenarioAttributeDialog(QDialog):
    def __init__(self, scenario_type, parent=None, font_size=14):
        super().__init__(parent)
        self.setWindowTitle(scenario_type)

        self.scenario_attributes = []  # List to store scenario attributes

        self.scenario_type = scenario_type
        self.font_size = font_size

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.left_layout = QVBoxLayout()  # Layout for labels
        self.right_layout = QVBoxLayout()  # Layout for input widgets

        # Add left and right layouts to a horizontal layout
        vlayout = QHBoxLayout()
        self.main_layout.addLayout(vlayout)
        vlayout.addLayout(self.left_layout)
        vlayout.addLayout(self.right_layout)

        # Add input widgets for each scenario attribute
        for scenario_type in config.SCENARIO_TYPES[scenario_type]:
            attribute = scenario_type[0]
            attr_type = scenario_type[1]
            default_value = None if len(scenario_type) == 2 else scenario_type[2]
            self.add_input_widget(attribute, attr_type, default_value)

        # Add select button
        button_layout = QHBoxLayout()
        select_button = QPushButton("Select")
        select_button.clicked.connect(self.set_attributes_before_closing)
        button_layout.addWidget(select_button)
        self.main_layout.addLayout(button_layout)

        # Resize and center the dialog
        self.resize(self.sizeHint().width(), self.sizeHint().height())
        self.center()

        self.setModal(True)
        self.exec_()

    def set_attributes_before_closing(self):
        """
        Set the scenario attributes based on the user input
        before closing the dialog.
        """
        for i in range(len(self.scenario_attributes)):
            attribute, attr_type, input_widgets = self.scenario_attributes[i]

            if attr_type in ("bool", "value"):
                line_edit = input_widgets
                try:
                    value = int(line_edit.text())
                    self.scenario_attributes[i] = (attribute, attr_type, value)
                except ValueError:
                    self.scenario_attributes[i] = None

            elif attr_type == "transform":
                pass  # Not used in any scenario currently

            elif "location" in attr_type:
                pass  # Must be done in the main window directly

            elif attr_type == "interval":
                try:
                    line_edit_from, line_edit_to = input_widgets
                    from_value = int(line_edit_from.text())
                    to_value = int(line_edit_to.text())
                    self.scenario_attributes[i] = (attribute, attr_type, [from_value, to_value])
                except ValueError:
                    self.scenario_attributes[i] = None

            elif attr_type == "choice":
                try:
                    combo_box = input_widgets
                    direction = combo_box.currentText()
                    self.scenario_attributes[i] = (attribute, attr_type, direction)
                except:
                    self.scenario_attributes[i] = None

            else:
                raise NotImplementedError(f"Type {attr_type} is not implemented yet")

        # Remove None values from the scenario_attributes list
        self.scenario_attributes = [attr for attr in self.scenario_attributes if attr is not None]

        self.close()

    def center(self):
        """
        Center the dialog on the screen.
        """
        frame_geometry = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def add_input_widget(self, attribute, attr_type, default_value):
        """
        Add input widgets for the given attribute and type.
        """
        if attr_type in ("bool", "value"):

            label = QLabel(f"{attribute.upper()}: ")
            self.left_layout.addWidget(label)

            font = QFont("Arial", self.font_size)
            label.setFont(font)

            line_edit = QLineEdit()
            line_edit.setPlaceholderText(str(default_value))
            line_edit.setAlignment(Qt.AlignCenter)
            line_edit.setValidator(QIntValidator())
            self.right_layout.addWidget(line_edit)
            line_edit.setFont(font)

            self.scenario_attributes.append((attribute, attr_type, line_edit))

        elif attr_type == "transform":
            pass  # Not used in any scenario currently

        elif "location" in attr_type:
            pass  # Must be done in the main window directly

        elif attr_type == "interval":
            font = QFont("Arial", self.font_size)

            label = QLabel(f"{attribute.upper()}: ")
            self.left_layout.addWidget(label)
            label.setFont(font)

            h_layout = QHBoxLayout()
            self.right_layout.addLayout(h_layout)

            line_edit_from = QLineEdit()
            line_edit_from.setPlaceholderText(str(default_value[0]))
            line_edit_from.setFont(font)
            line_edit_from.setValidator(QIntValidator())
            line_edit_from.setAlignment(Qt.AlignCenter)
            h_layout.addWidget(line_edit_from)

            h_layout.addWidget(QLabel("-"))

            line_edit_to = QLineEdit()
            line_edit_to.setPlaceholderText(str(default_value[1]))
            line_edit_to.setFont(font)
            line_edit_to.setAlignment(Qt.AlignCenter)
            line_edit_to.setValidator(QIntValidator())
            h_layout.addWidget(line_edit_to)

            self.scenario_attributes.append((attribute, attr_type, (line_edit_from, line_edit_to)))

        elif attr_type == "choice":
            font = QFont("Arial", self.font_size)

            label = QLabel(f"{attribute.upper()}: ")
            self.left_layout.addWidget(label)
            label.setFont(font)

            combo_box = QComboBox(self)
            combo_box.setFont(font)
            combo_box.addItem("left")
            combo_box.addItem("right")
            self.right_layout.addWidget(combo_box)

            self.scenario_attributes.append((attribute, attr_type, combo_box))

        else:
            raise NotImplementedError(f"Type {attr_type} is not implemented yet")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    scenario_type = "SignalizedJunctionLeftTurn"
    scenario_attribute_dialog = ScenarioAttributeDialog(scenario_type)
    print(scenario_attribute_dialog.scenario_attributes)
    sys.exit(app.exec_())
