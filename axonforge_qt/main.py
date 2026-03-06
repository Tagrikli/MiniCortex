"""
main.py — Entry point for the PySide6 AxonForge Node Editor.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase

from .bridge import BridgeAPI
from .main_window import MainWindow


def load_stylesheet() -> str:
    """Load the QSS theme file."""
    qss_path = Path(__file__).parent / "themes" / "dark.qss"
    if qss_path.exists():
        return qss_path.read_text()
    return ""


def main() -> None:
    print("AxonForge — PySide6 Node Editor")
    print("=" * 40)

    app = QApplication(sys.argv)

    # Load stylesheet
    qss = load_stylesheet()
    if qss:
        app.setStyleSheet(qss)

    # Initialise bridge (discovers nodes, builds palette)
    bridge = BridgeAPI()
    bridge.init()

    # Create and show main window
    window = MainWindow(bridge, ui_fps=60.0)
    window.show()

    print("AxonForge ready.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
