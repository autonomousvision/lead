"""
This script creates a GUI window using PyQt5 that displays a loading animation and a text message
while performing a time-consuming operation. The operation can be either a blocking function or a
non-blocking function (e.g., a function that emits signals or uses callbacks). This design prevents
the main GUI thread from freezing during the operation, ensuring a smooth user experience.
"""

import sys
import time

from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEventLoop
from PyQt5.QtGui import QFont, QMovie


class LongRunningTask(QThread):
    """
    A QThread subclass that runs a time-consuming task in a separate thread.
    """

    task_completed = pyqtSignal(object)

    def __init__(self, parent, task_function):
        """
        Initializes the LongRunningTask thread.

        Args:
            parent (QObject): The parent object of the thread.
            task_function (callable): The function to be executed in the separate thread.
        """
        super().__init__(parent)
        self.task_function = task_function

    def run(self):
        """
        Runs the task function and emits the task_completed signal when done.
        """
        time.sleep(0.5)
        result = self.task_function()
        self.task_completed.emit(result)


class LoadingIndicatorWindow(QDialog):
    def __init__(self, parent, message_text, task_function):
        """
        Initializes the LoadingIndicatorWindow.

        Args:
            message_text (str): The text message to display alongside the loading animation.
            task_function (callable): The time-consuming task function to be executed.
            parent (QWidget, optional): The parent widget of the window. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Loading...")
        self.setModal(True)

        self.task_function = task_function

        # Create a QMovie object from the GIF file
        self.loading_animation = QMovie("scripts/images/loading_animation.gif")
        self.animation_label = QLabel()
        self.animation_label.setAlignment(Qt.AlignCenter)
        self.animation_label.setMovie(self.loading_animation)
        self.loading_animation.start()

        # Create a QLabel for the text message
        self.message_label = QLabel()
        self.message_label.setText(message_text)
        self.message_label.setAlignment(Qt.AlignCenter)

        # Set a larger font for the text message
        font = QFont()
        font.setPointSize(20)
        self.message_label.setFont(font)

        # Create a vertical layout and add the labels
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.message_label)
        main_layout.addWidget(self.animation_label)
        self.setLayout(main_layout)

        # Remove the window frame for a cleaner look
        self.setWindowFlag(Qt.FramelessWindowHint)

        # Start the long-running task in a separate thread
        self.task_thread = LongRunningTask(self, self.task_function)
        self.task_thread.task_completed.connect(self.close)

        self.show()
        self.task_thread.start()

        # Create an event loop for the loading window
        self.event_loop = QEventLoop()
        self.event_loop.exec_()

    def closeEvent(self, event):
        """
        Overrides the closeEvent method to prevent the user from closing the window manually.

        Args:
            event (QCloseEvent): The close event object.
        """
        # Ignore the close event unless it's a spontaneous event
        if not event.spontaneous():
            self.event_loop.quit()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    # Create the Qt application
    app = QApplication(sys.argv)

    # Create and show the LoadingIndicatorWindow
    import carla

    client = carla.Client()
    loading_window = LoadingIndicatorWindow(None, "Loading map...", lambda: client.load_world("Town01"))

    # Start the Qt event loop
    sys.exit(app.exec_())
