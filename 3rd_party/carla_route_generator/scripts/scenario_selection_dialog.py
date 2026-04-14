import sys
from PyQt5.QtWidgets import (
    QWidget,
    QScrollBar,
    QGridLayout,
    QLineEdit,
    QLabel,
    QDialog,
    QApplication,
    QPushButton,
    QVBoxLayout,
    QScrollArea,
)
import config


class ScenarioSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scenario Selection")
        self.setGeometry(100, 100, 400, 600)

        self.selected_scenario = None

        # Get scenario types from the config
        self.SCENARIO_TYPES = config.SCENARIO_TYPES

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Label for the filter
        filter_label = QLabel("Filter")
        main_layout.addWidget(filter_label)

        # Text field to filter scenarios
        self.filter_text_field = QLineEdit()
        self.filter_text_field.textChanged.connect(self.filter_available_scenarios)
        main_layout.addWidget(self.filter_text_field)

        # Create a grid layout inside the scroll area
        self.grid_layout = QGridLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_widget.setLayout(self.grid_layout)

        scroll_area.setWidget(scroll_widget)

        scrollbar = QScrollBar()
        scrollbar.setOrientation(1)  # Vertical orientation
        scrollbar.setMaximum(100)  # Set the maximum value
        main_layout.addWidget(scrollbar)

        # Add scenario labels and buttons to the grid layou
        self.list = []
        self.update_scenario_list(self.SCENARIO_TYPES)

        self.setModal(True)
        self.exec_()

    def update_scenario_list(self, scenario_types):
        for i, scenario in enumerate(sorted(scenario_types.keys())):
            scenario_label = QLabel(scenario)
            select_button = QPushButton(">")
            select_button.setFixedWidth(select_button.minimumSizeHint().width())
            select_button.setMinimumHeight(
                select_button.minimumSizeHint().height()
            )  # Set minimum height to preferred height
            select_button.clicked.connect(lambda _, scenario=scenario: self.on_scenario_selected(scenario))

            self.list.append(select_button)
            self.list.append(scenario_label)

            self.grid_layout.addWidget(scenario_label, i, 0)
            self.grid_layout.addWidget(select_button, i, 1)

    def filter_available_scenarios(self, text):
        scenario_types = {key: value for (key, value) in self.SCENARIO_TYPES.items() if text.lower() in key.lower()}
        for item in self.list:
            self.grid_layout.removeWidget(item)
            # Delete the button from memory
            item.deleteLater()

        self.list.clear()
        self.update_scenario_list(scenario_types)

    def on_scenario_selected(self, scenario):
        self.selected_scenario = scenario
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ScenarioSelectionDialog()
    sys.exit(app.exec_())
