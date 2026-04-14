"""
This script creates a GUI window using PyQt5 that displays a list of available maps from a Carla client.
The user can select a map from the list, and the selected map's name is stored in the `selected_map_name` attribute.
"""

import sys
from PyQt5.QtWidgets import QDialog, QListWidgetItem, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QListWidget
from PyQt5.QtCore import Qt
from carla_simulator_client import CarlaClient


class MapSelectionDialog(QDialog):
    def __init__(self, carla_client, parent=None):
        """
        Initializes the MapSelectionDialog.

        Args:
            carla_client (CarlaClient): An instance of the CarlaClient class.
            parent (QWidget, optional): The parent widget of the window. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Route Creator")
        self.setGeometry(100, 100, 500, 300)
        self.carla_client = carla_client

        # Create the main layout and set it for the window
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Get the list of available maps from the Carla client and sort it
        available_maps = sorted(carla_client.get_available_maps())

        # Create a list widget to display the available maps
        maps_list_widget = QListWidget()
        for map_name in available_maps:
            map_item = QListWidgetItem(map_name)
            map_item.setTextAlignment(Qt.AlignHCenter)
            maps_list_widget.addItem(map_item)

        # Set the first item in the list as the initially selected map
        first_map_item = maps_list_widget.item(0)
        maps_list_widget.setCurrentItem(first_map_item)
        self.selected_map_name = first_map_item.text()

        # Connect the itemClicked signal to handle map selection
        maps_list_widget.itemClicked.connect(self.handle_map_selection)
        main_layout.addWidget(maps_list_widget)

        # Create a layout for the buttons
        button_layout = QHBoxLayout()

        # Create the "Select" button
        select_button = QPushButton("Select")
        select_button.clicked.connect(self.close)
        button_layout.addWidget(select_button)

        # Create the "Cancel" button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.handle_cancel_button_click)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        self.setModal(True)
        self.exec_()

    def closeEvent(self, event):
        """
        Handles the close event of the window.

        Args:
            event (QEvent): The close event object.
        """
        if event.spontaneous():
            self.selected_map_name = None
        event.accept()

    def handle_cancel_button_click(self):
        """
        Handles the "Cancel" button click event.
        Sets the `selected_map_name` to None and closes the window.
        """
        self.selected_map_name = None
        self.close()

    def handle_map_selection(self, selected_item):
        """
        Handles the map selection event.

        Args:
            selected_item (QListWidgetItem): The selected map item.
        """
        self.selected_map_name = selected_item.text()


if __name__ == "__main__":
    # Create an instance of the CarlaClient
    carla_client = CarlaClient("localhost", 2000)

    # Create the Qt application
    app = QApplication(sys.argv)

    # Create and show the MapSelectionDialog
    window = MapSelectionDialog(carla_client)
    window.show()

    # Start the Qt event loop
    sys.exit(app.exec_())
