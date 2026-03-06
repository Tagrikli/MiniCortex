"""
SaveWorkspaceDialog — Modal dialog for entering a workspace name.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
)


class SaveWorkspaceDialog(QDialog):
    """Simple dialog to enter a workspace name."""

    def __init__(self, current_name: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("save_dialog")
        self.setWindowTitle("Save Workspace")
        self.setFixedSize(340, 140)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Save Workspace")
        layout.addWidget(title)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Enter workspace name...")
        self._input.setText(current_name)
        layout.addWidget(self._input)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setObjectName("save_btn")
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

        self._input.returnPressed.connect(self.accept)
        self._input.setFocus()

    @property
    def workspace_name(self) -> str:
        return self._input.text().strip()
