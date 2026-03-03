"""
MainWindow — QMainWindow assembling header, palette, canvas, and console.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QSizePolicy, QMessageBox,
)

from .bridge import BridgeAPI
from .computation_thread import ComputationThread
from .canvas.editor_scene import EditorScene
from .canvas.editor_view import EditorView
from .panels.palette import PalettePanel
from .panels.console import ConsolePanel
from .dialogs.save_workspace import SaveWorkspaceDialog


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, bridge: BridgeAPI, ui_fps: float = 30.0) -> None:
        super().__init__()
        self.bridge = bridge
        self._ui_fps = ui_fps

        self.setWindowTitle("MiniCortex — Node Editor")
        self.setMinimumSize(1200, 700)

        # ── Central widget ───────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────────────
        self._header_bar = self._build_header()
        root_layout.addWidget(self._header_bar)

        # ── Body: palette + canvas ───────────────────────────────────────
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._palette = PalettePanel(bridge)
        body_layout.addWidget(self._palette)

        self._scene = EditorScene(bridge)
        self._view = EditorView(self._scene)
        body_layout.addWidget(self._view)

        # Make palette resizable using splitter
        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._palette)
        splitter.addWidget(self._view)
        splitter.setSizes([200, 800])
        splitter.setStretchFactor(0, 0)  # palette doesn't stretch
        splitter.setStretchFactor(1, 1)  # view stretches
        # Replace the layout widget with splitter
        body_layout.removeWidget(self._palette)
        body_layout.removeWidget(self._view)
        body_layout.addWidget(splitter)

        root_layout.addWidget(body)

        # ── Console panel ────────────────────────────────────────────────
        self._console = ConsolePanel(bridge)
        self._console.setVisible(True)
        root_layout.addWidget(self._console)

        # ── Computation thread ───────────────────────────────────────────
        self._comp_thread = ComputationThread(bridge)
        self._comp_thread.network_error.connect(self._on_network_error)
        self._comp_thread.start()

        # ── UI refresh timer ─────────────────────────────────────────────
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._on_refresh_tick)
        self._refresh_timer.start(int(1000.0 / self._ui_fps))

        # ── Keyboard shortcuts ───────────────────────────────────────────
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_workspace)
        QShortcut(QKeySequence("Ctrl+Space"), self, self._toggle_network)
        QShortcut(QKeySequence("Ctrl+Return"), self, self._step_network)
        QShortcut(QKeySequence("Shift+T"), self, self._toggle_console)

        # ── Load initial topology ────────────────────────────────────────
        self._load_initial_state()

    # ── Header ───────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header_bar")
        header.setFixedHeight(44)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        # Title
        title = QLabel("MiniCortex")
        title.setObjectName("app_title")
        layout.addWidget(title)

        # Workspace combo
        self._workspace_combo = QComboBox()
        self._workspace_combo.setObjectName("workspace_combo")
        self._workspace_combo.setEditable(False)
        self._workspace_combo.addItem("Unsaved Workspace")
        self._workspace_combo.activated.connect(self._on_workspace_selected)
        layout.addWidget(self._workspace_combo)

        # New workspace button
        new_btn = QPushButton("+")
        new_btn.setObjectName("new_workspace_btn")
        new_btn.setFixedSize(24, 24)
        new_btn.setToolTip("New Workspace")
        new_btn.clicked.connect(self._new_workspace)
        layout.addWidget(new_btn)

        layout.addStretch()

        # Network controls
        self._toggle_btn = QPushButton("Start")
        self._toggle_btn.setObjectName("network_toggle")
        self._toggle_btn.clicked.connect(self._toggle_network)
        layout.addWidget(self._toggle_btn)

        self._iterate_btn = QPushButton("Iterate")
        self._iterate_btn.setObjectName("network_iterate")
        self._iterate_btn.setToolTip("Execute one step")
        self._iterate_btn.clicked.connect(self._step_network)
        layout.addWidget(self._iterate_btn)

        speed_label = QLabel("Max Hz")
        speed_label.setObjectName("speed_label")
        layout.addWidget(speed_label)

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setObjectName("network_speed_slider")
        self._speed_slider.setMinimum(1)
        self._speed_slider.setMaximum(300)
        self._speed_slider.setValue(60)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_slider)

        self._speed_value = QLabel("60")
        self._speed_value.setObjectName("speed_value")
        layout.addWidget(self._speed_value)

        self._actual_hz = QLabel("0.0 Hz")
        self._actual_hz.setObjectName("actual_hz")
        layout.addWidget(self._actual_hz)

        return header

    # ── Initial state ────────────────────────────────────────────────────

    def _load_initial_state(self) -> None:
        topology = self.bridge.get_topology()
        self._scene.rebuild_from_topology(topology)
        self._refresh_workspace_list()

    def _refresh_workspace_list(self) -> None:
        self._workspace_combo.clear()
        self._workspace_combo.addItem("Unsaved Workspace")
        for ws in self.bridge.list_workspaces():
            self._workspace_combo.addItem(ws["name"])
        if self.bridge.current_workspace:
            idx = self._workspace_combo.findText(self.bridge.current_workspace)
            if idx >= 0:
                self._workspace_combo.setCurrentIndex(idx)

    # ── UI refresh ───────────────────────────────────────────────────────

    def _on_refresh_tick(self) -> None:
        buf = self.bridge.read_display_buffer()
        if buf is not None:
            self._scene.update_displays(buf)

        # Update error states on nodes
        failed_nodes = self.bridge.network.failed_nodes
        for node_item in self._scene._node_items.values():
            node_item.set_error_state(node_item.node_id in failed_nodes)

        # Update network state UI
        state = self.bridge.get_network_state()
        running = state["running"]
        self._toggle_btn.setText("Stop" if running else "Start")
        self._toggle_btn.setProperty("running", "true" if running else "false")
        self._toggle_btn.style().unpolish(self._toggle_btn)
        self._toggle_btn.style().polish(self._toggle_btn)
        self._actual_hz.setText(f"{state['actual_hz']:.1f} Hz")

        # Update header bar style based on running state
        self._header_bar.setProperty("running", "true" if running else "false")
        self._header_bar.style().unpolish(self._header_bar)
        self._header_bar.style().polish(self._header_bar)

    # ── Network controls ─────────────────────────────────────────────────

    def _toggle_network(self) -> None:
        if self.bridge.network.running:
            self.bridge.stop_network()
        else:
            # Clear failed nodes when starting fresh
            self.bridge.network.clear_failed_nodes()
            self.bridge.start_network()

    def _step_network(self) -> None:
        self.bridge.step_network()

    def _on_speed_changed(self, value: int) -> None:
        self._speed_value.setText(str(value))
        self.bridge.set_network_speed(float(value))

    def _on_network_error(self, node_id: str, node_name: str, error: str, tb: str) -> None:
        error_msg = f"Network error in '{node_name}': {error}"
        print(error_msg)
        print(tb)
        # Also log to console panel if available
        if hasattr(self, '_console'):
            self._console.append_message(error_msg, is_error=True)
            self._console.append_message(tb, is_error=True)
        # Ensure the node shows error state immediately
        node_item = self._scene._node_items.get(node_id)
        if node_item:
            node_item.set_error_state(True)

    def _toggle_console(self) -> None:
        """Toggle console panel visibility with Shift+T shortcut."""
        if hasattr(self, '_console'):
            self._console.set_visible(not self._console.is_visible())

    # ── Workspace ────────────────────────────────────────────────────────

    def _on_workspace_selected(self, index: int) -> None:
        if index == 0:
            return  # "Unsaved Workspace"
        name = self._workspace_combo.itemText(index)
        try:
            viewport = self.bridge.load_workspace(name)
            topology = self.bridge.get_topology()
            self._scene.rebuild_from_topology(topology)
        except Exception as e:
            print(f"Failed to load workspace: {e}")

    def _save_workspace(self) -> None:
        if self.bridge.current_workspace:
            self.bridge.save_workspace(self.bridge.current_workspace)
            return
        dlg = SaveWorkspaceDialog(parent=self)
        if dlg.exec() and dlg.workspace_name:
            self.bridge.save_workspace(dlg.workspace_name)
            self._refresh_workspace_list()

    def _new_workspace(self) -> None:
        self.bridge.clear_workspace()
        topology = self.bridge.get_topology()
        self._scene.rebuild_from_topology(topology)
        self._refresh_workspace_list()

    # ── Cleanup ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._refresh_timer.stop()
        self._comp_thread.request_stop()
        self._comp_thread.wait(2000)
        super().closeEvent(event)
