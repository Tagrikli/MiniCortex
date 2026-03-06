"""
PortItem — Small coloured square on the edge of a node representing an I/O port.
"""

from typing import List, Optional

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter
from PySide6.QtWidgets import QGraphicsItem

from .z_layers import Z_PORT, Z_PORT_HIT

# Theme colours
INPUT_PORT_COLOR = QColor("#3dff6f")
OUTPUT_PORT_COLOR = QColor("#ff4dff")
INPUT_PORT_BG = QColor("#1a6638")  # Solid dark green
OUTPUT_PORT_BG = QColor("#661a66")  # Solid dark magenta
PORT_SIZE = 8.0
PORT_SIZE_HIGHLIGHTED = 12.0


class PortItem(QGraphicsItem):
    """A connection port drawn as a small square on the node edge."""

    def __init__(
        self,
        node_id: str,
        port_name: str,
        port_type: str,       # "input" or "output"
        data_type: str = "any",
        data_types: Optional[List[str]] = None,  # List of types for multi-type ports
        label: str = "",
        parent: QGraphicsItem = None,
    ) -> None:
        super().__init__(parent)
        self.node_id = node_id
        self.port_name = port_name
        self.port_type = port_type
        self.data_type = data_type
        self.data_types = data_types  # List of types for tooltip display
        self.label = label
        self._connected = False

        self.setAcceptHoverEvents(True)
        self._hovered = False
        self._highlighted = False  # Highlighted via snap radius or label hover

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setZValue(Z_PORT)

    # ── Geometry ─────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        size = PORT_SIZE_HIGHLIGHTED if (self._highlighted or self._hovered) else PORT_SIZE
        return QRectF(-size / 2, -size / 2, size, size)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        is_input = self.port_type == "input"
        border_color = INPUT_PORT_COLOR if is_input else OUTPUT_PORT_COLOR

        if self._connected or self._hovered or self._highlighted:
            fill = border_color
        else:
            fill = INPUT_PORT_BG if is_input else OUTPUT_PORT_BG

        painter.setPen(QPen(border_color, 1))
        painter.setBrush(QBrush(fill))
        r = self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRect(r)

    # ── State ────────────────────────────────────────────────────────────

    def set_connected(self, connected: bool) -> None:
        if self._connected != connected:
            self._connected = connected
            self.update()

    def set_highlighted(self, highlighted: bool) -> None:
        """Set highlight state (from snap proximity or label hover)."""
        if self._highlighted != highlighted:
            self._highlighted = highlighted
            self.update()

    @property
    def center_scene_pos(self) -> QPointF:
        """Centre of the port in scene coordinates."""
        return self.mapToScene(QPointF(0, 0))

    # ── Hover ────────────────────────────────────────────────────────────

    def hoverEnterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)


class PortHitZoneItem(QGraphicsItem):
    """Transparent per-port hit area covering port + label corridor."""

    def __init__(self, port: PortItem, parent: QGraphicsItem = None) -> None:
        super().__init__(parent)
        self.port = port
        self._rect = QRectF()
        self.setZValue(Z_PORT_HIT)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setAcceptHoverEvents(False)

    def set_rect(self, rect: QRectF) -> None:
        if rect == self._rect:
            return
        self.prepareGeometryChange()
        self._rect = QRectF(rect)

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        # Intentionally transparent; this item exists only for hit-testing.
        return
