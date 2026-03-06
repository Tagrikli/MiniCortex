"""
ConnectionItem — Bezier curve connecting two ports.
"""

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPen, QColor, QPainterPath
from PySide6.QtWidgets import QGraphicsPathItem

from .z_layers import Z_CONNECTION, Z_OVERLAY

CONN_COLOR = QColor("#26324d")
CONN_HOVER_COLOR = QColor("#00f5ff")
CONN_WIDTH = 2.0


class ConnectionItem(QGraphicsPathItem):
    """A bezier connection between two ports."""

    def __init__(self, conn_data: dict, parent=None) -> None:
        super().__init__(parent)
        self.conn_data = conn_data  # {from_node, from_output, to_node, to_input}

        self.setPen(QPen(CONN_COLOR, CONN_WIDTH))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(Z_CONNECTION)

    def update_path(self, start: QPointF, end: QPointF) -> None:
        """Recompute the bezier path between two points."""
        path = QPainterPath(start)
        dx = abs(end.x() - start.x())
        offset = max(50.0, dx * 0.5)
        c1 = QPointF(start.x() + offset, start.y())
        c2 = QPointF(end.x() - offset, end.y())
        path.cubicTo(c1, c2, end)
        self.setPath(path)

    # ── Hover ────────────────────────────────────────────────────────────

    def hoverEnterEvent(self, event) -> None:
        self.setPen(QPen(CONN_HOVER_COLOR, CONN_WIDTH))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setPen(QPen(CONN_COLOR, CONN_WIDTH))
        super().hoverLeaveEvent(event)

    # ── Shape for hit testing ────────────────────────────────────────────

    def shape(self):
        """Wider hit area for easier clicking."""
        from PySide6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)
        return stroker.createStroke(self.path())


class ConnectionPreviewItem(QGraphicsPathItem):
    """Temporary dashed line shown while dragging a new connection."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        pen = QPen(CONN_HOVER_COLOR, CONN_WIDTH)
        pen.setDashPattern([5, 5])
        self.setPen(pen)
        self.setZValue(Z_OVERLAY)

    def update_path(self, start: QPointF, end: QPointF) -> None:
        path = QPainterPath(start)
        dx = abs(end.x() - start.x())
        offset = max(50.0, dx * 0.5)
        c1 = QPointF(start.x() + offset, start.y())
        c2 = QPointF(end.x() - offset, end.y())
        path.cubicTo(c1, c2, end)
        self.setPath(path)
