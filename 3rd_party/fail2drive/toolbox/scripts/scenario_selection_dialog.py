import sys
from PyQt6.QtWidgets import (
    QWidget,
    QGridLayout,
    QLineEdit,
    QLabel,
    QDialog,
    QApplication,
    QPushButton,
    QVBoxLayout,
    QScrollArea,
    QFrame,
)
import config
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


SCENARIO_GROUPS = [
    ("Fail2Drive", [
        "BadParkingObstacle",
        "BadParkingObstacleTwoWays",
        "ConstructionObstacleOppositeLane",
        "ConstructionObstaclePedestrian",
        "ConstructionObstacleRightLane",
        "CustomObstacle",
        "CustomObstacleTwoWays",
        "HardBrakeNoLights",
        "ImageOnObject",
        "NormalVehicleRunningRedLight",
        "NormalVehicleTakingPriority",
        "ObscuredStopSign",
        "PedestrianCrowd",
        "PedestriansOnRoad",
        "PermutedConstructionObstacle",
        "PermutedConstructionObstacleTwoWays",
        "RoadBlocked",
    ]),
    ("Junctions", [
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "OppositeVehicleRunningRedLight",
        "OppositeVehicleTakingPriority",
        "BlockedIntersection",
        "PriorityAtJunction",
    ]),
    ("Crossing Actors", [
        "DynamicObjectCrossing",
        "ParkingCrossingPedestrian",
        "PedestrianCrossing",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
        "CrossingBicycleFlow",
    ]),
    ("Actor Flows & Merging", [
        "EnterActorFlow",
        "EnterActorFlowV2",
        "InterurbanActorFlow",
        "InterurbanAdvancedActorFlow",
        "HighwayExit",
        "MergerIntoSlowTraffic",
        "MergerIntoSlowTrafficV2",
        "HighwayCutIn",
        "ParkingCutIn",
        "StaticCutIn",
    ]),
    ("Route Obstacles", [
        "ConstructionObstacle",
        "ConstructionObstacleTwoWays",
        "Accident",
        "AccidentTwoWays",
        "ParkedObstacle",
        "ParkedObstacleTwoWays",
        "VehicleOpensDoorTwoWays",
        "HazardAtSideLane",
        "HazardAtSideLaneTwoWays",
        "InvadingTurn",
    ]),
    ("Other", [
        "ControlLoss",
        "HardBreakRoute",
        "ParkingExit",
        "YieldToEmergencyVehicle",
        "BackgroundActivityParametrizer",
    ]),
]


class ScenarioSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scenario Selection")
        self.setGeometry(100, 100, 400, 600)

        self.selected_scenario = None
        self.SCENARIO_TYPES = config.SCENARIO_TYPES

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.filter_text_field = QLineEdit()
        self.filter_text_field.setPlaceholderText("Filter...")
        self.filter_text_field.textChanged.connect(self.filter_available_scenarios)
        main_layout.addWidget(self.filter_text_field)

        self.grid_layout = QGridLayout()
        self.grid_layout.setVerticalSpacing(2)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_widget.setLayout(self.grid_layout)
        scroll_area.setWidget(scroll_widget)

        self._widgets = []
        self._populate_grouped()

        self.setModal(True)
        self.exec()

    def _clear_grid(self):
        for w in self._widgets:
            self.grid_layout.removeWidget(w)
            w.deleteLater()
        self._widgets.clear()

    def _add_scenario_row(self, row, scenario):
        tooltip = config.get_scenario_tooltip(scenario)
        label = QLabel(scenario)
        label.setToolTip(tooltip)
        btn = QPushButton(">")
        btn.setFixedWidth(btn.minimumSizeHint().width())
        btn.setMinimumHeight(btn.minimumSizeHint().height())
        btn.setToolTip(tooltip)
        btn.clicked.connect(lambda _, s=scenario: self.on_scenario_selected(s))
        self.grid_layout.addWidget(label, row, 0)
        self.grid_layout.addWidget(btn, row, 1)
        self._widgets += [label, btn]
        return row + 1

    def _add_group_header(self, row, title):
        header = QLabel(title)
        font = QFont()
        font.setBold(True)
        header.setFont(font)
        header.setStyleSheet("color: #5a6a7a; padding-top: 6px;")
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.grid_layout.addWidget(header, row, 0, 1, 2)
        self.grid_layout.addWidget(separator, row + 1, 0, 1, 2)
        self._widgets += [header, separator]
        return row + 2

    def _populate_grouped(self):
        row = 0
        for group_name, scenarios in SCENARIO_GROUPS:
            present = [s for s in scenarios if s in self.SCENARIO_TYPES]
            if not present:
                continue
            row = self._add_group_header(row, group_name)
            for scenario in present:
                row = self._add_scenario_row(row, scenario)

    def _populate_filtered(self, text):
        matches = sorted(k for k in self.SCENARIO_TYPES if text.lower() in k.lower())
        row = 0
        for scenario in matches:
            row = self._add_scenario_row(row, scenario)

    def filter_available_scenarios(self, text):
        self._clear_grid()
        if text.strip():
            self._populate_filtered(text)
        else:
            self._populate_grouped()

    def on_scenario_selected(self, scenario):
        self.selected_scenario = scenario
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ScenarioSelectionDialog()
    sys.exit(app.exec())
