"""
EditorScene — QGraphicsScene that manages nodes, connections, and interaction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QLineF
from PySide6.QtGui import QPen, QColor
from PySide6.QtWidgets import QGraphicsScene, QGraphicsSceneMouseEvent, QGraphicsProxyWidget

from .node_item import NodeItem
from .port_item import PortItem
from .connection_item import ConnectionItem, ConnectionPreviewItem

if TYPE_CHECKING:
    from ..bridge import BridgeAPI


class EditorScene(QGraphicsScene):
    """Scene managing all node and connection items."""

    # Signals
    node_selected = Signal(str)          # node_id
    connection_selected = Signal(dict)   # conn_data
    selection_cleared = Signal()

    PORT_SNAP_DISTANCE = 30.0  # pixels for proximity-based port selection

    def __init__(self, bridge: BridgeAPI, parent=None) -> None:
        super().__init__(parent)
        self.bridge = bridge

        self._node_items: Dict[str, NodeItem] = {}
        self._connection_items: List[ConnectionItem] = []

        # Connection drag state
        self._connecting = False
        self._connect_from_port: Optional[PortItem] = None
        self._preview_line: Optional[ConnectionPreviewItem] = None

        self._selected_connection: Optional[ConnectionItem] = None
        self._highlighted_port: Optional[PortItem] = None

    # ── Node management ──────────────────────────────────────────────────

    def add_node_item(self, schema: dict) -> NodeItem:
        item = NodeItem(schema)
        self.addItem(item)
        self._node_items[schema["node_id"]] = item

        # Wire signals
        item.signals.position_changed.connect(self._on_node_moved)
        item.signals.property_changed.connect(self._on_property_changed)
        item.signals.action_triggered.connect(self._on_action_triggered)
        item.signals.reload_requested.connect(self._on_reload_requested)

        return item

    def remove_node_item(self, node_id: str) -> None:
        item = self._node_items.pop(node_id, None)
        if item:
            self.removeItem(item)
        # Remove associated connections
        i = 0
        while i < len(self._connection_items):
            c = self._connection_items[i]
            if c.conn_data["from_node"] == node_id or c.conn_data["to_node"] == node_id:
                self.removeItem(c)
                self._connection_items.pop(i)
            else:
                i += 1
        self._update_port_connected_states()

    def get_node_item(self, node_id: str) -> Optional[NodeItem]:
        return self._node_items.get(node_id)

    # ── Connection management ────────────────────────────────────────────

    def add_connection_item(self, conn_data: dict) -> ConnectionItem:
        item = ConnectionItem(conn_data)
        self.addItem(item)
        self._connection_items.append(item)
        self._update_connection_path(item)
        self._update_port_connected_states()
        return item

    def remove_connection_item(self, conn_data: dict) -> None:
        for i, item in enumerate(self._connection_items):
            d = item.conn_data
            if (d["from_node"] == conn_data["from_node"] and
                d["from_output"] == conn_data["from_output"] and
                d["to_node"] == conn_data["to_node"] and
                d["to_input"] == conn_data["to_input"]):
                self.removeItem(item)
                self._connection_items.pop(i)
                break
        self._update_port_connected_states()

    def _update_connection_path(self, item: ConnectionItem) -> None:
        d = item.conn_data
        from_node = self._node_items.get(d["from_node"])
        to_node = self._node_items.get(d["to_node"])
        if not from_node or not to_node:
            return
        from_port = from_node.get_port(d["from_output"], "output")
        to_port = to_node.get_port(d["to_input"], "input")
        if from_port and to_port:
            item.update_path(from_port.center_scene_pos, to_port.center_scene_pos)

    def update_connections_for_node(self, node_id: str) -> None:
        """Called when a node moves — update all its connections."""
        for item in self._connection_items:
            d = item.conn_data
            if d["from_node"] == node_id or d["to_node"] == node_id:
                self._update_connection_path(item)

    def update_all_connections(self) -> None:
        for item in self._connection_items:
            self._update_connection_path(item)

    def _update_port_connected_states(self) -> None:
        """Update connected visual state on all ports."""
        connected_ports = set()
        for item in self._connection_items:
            d = item.conn_data
            connected_ports.add((d["from_node"], d["from_output"], "output"))
            connected_ports.add((d["to_node"], d["to_input"], "input"))

        for node_item in self._node_items.values():
            for port in node_item.input_ports:
                port.set_connected((port.node_id, port.port_name, "input") in connected_ports)
            for port in node_item.output_ports:
                port.set_connected((port.node_id, port.port_name, "output") in connected_ports)

    def _nearest_port_at(self, scene_pos: QPointF, max_distance: Optional[float] = None) -> Optional[PortItem]:
        """Return the nearest PortItem within max_distance, or None."""
        effective_dist = max_distance if max_distance is not None else self.PORT_SNAP_DISTANCE
        best = None
        best_dist = effective_dist
        for node_item in self._node_items.values():
            for port in node_item.input_ports + node_item.output_ports:
                port_center = port.center_scene_pos
                dist = QLineF(scene_pos, port_center).length()
                if dist <= best_dist:
                    best = port
                    best_dist = dist
        return best

    def _update_port_highlight(self, scene_pos: QPointF) -> None:
        """Update port highlighting based on snap proximity and label hover."""
        # Find the nearest port within snap distance or under label
        nearest_snap = self._nearest_port_at(scene_pos)
        label_port = self._port_at_label(scene_pos)
        
        # Determine which port to highlight
        new_highlight = nearest_snap or label_port
        
        # Don't highlight the source port (the one we're dragging from)
        if new_highlight and self._connect_from_port:
            if (new_highlight.node_id == self._connect_from_port.node_id and
                new_highlight.port_name == self._connect_from_port.port_name and
                new_highlight.port_type == self._connect_from_port.port_type):
                new_highlight = None
        
        # Clear previous highlight if different
        if self._highlighted_port and self._highlighted_port != new_highlight:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None
        
        # Set new highlight
        if new_highlight and new_highlight != self._highlighted_port:
            new_highlight.set_highlighted(True)
            self._highlighted_port = new_highlight

    def _port_at_label(self, scene_pos: QPointF) -> Optional[PortItem]:
        """Return the PortItem whose label contains scene_pos, or None."""
        HEADER_HEIGHT = 22.0
        PORT_ROW_HEIGHT = 18.0
        LABEL_LEFT = 12.0
        LABEL_RIGHT_MARGIN = 14.0

        for node_item in self._node_items.values():
            # Get node position in scene coordinates
            node_pos = node_item.pos()
            width = node_item._width
            half_width = width / 2

            # Check input port labels (left side)
            for i, port in enumerate(node_item.input_ports):
                label_x = node_pos.x() + LABEL_LEFT
                label_y = node_pos.y() + HEADER_HEIGHT + i * PORT_ROW_HEIGHT
                label_w = half_width - LABEL_LEFT - LABEL_RIGHT_MARGIN
                label_h = PORT_ROW_HEIGHT
                label_rect = QRectF(label_x, label_y, label_w, label_h)
                if label_rect.contains(scene_pos):
                    return port

            # Check output port labels (right side)
            for i, port in enumerate(node_item.output_ports):
                label_x = node_pos.x() + half_width + 2
                label_y = node_pos.y() + HEADER_HEIGHT + i * PORT_ROW_HEIGHT
                label_w = half_width - LABEL_RIGHT_MARGIN - 2
                label_h = PORT_ROW_HEIGHT
                label_rect = QRectF(label_x, label_y, label_w, label_h)
                if label_rect.contains(scene_pos):
                    return port

        return None

    # ── Full rebuild ─────────────────────────────────────────────────────

    def rebuild_from_topology(self, topology: dict) -> None:
        """Clear and rebuild all items from a topology snapshot."""
        self.clear()
        self._node_items.clear()
        self._connection_items.clear()
        self._preview_line = None
        self._connecting = False

        for schema in topology.get("nodes", []):
            self.add_node_item(schema)

        for conn in topology.get("connections", []):
            self.add_connection_item(conn)

    # ── Display updates ──────────────────────────────────────────────────

    def update_displays(self, display_buffer: Dict[str, Dict]) -> None:
        """Push display data to node items."""
        for node_id, outputs in display_buffer.items():
            item = self._node_items.get(node_id)
            if item:
                item.update_displays(outputs)

    # ── Connection drag ──────────────────────────────────────────────────

    def start_connection_drag(self, port: PortItem, scene_pos: QPointF) -> None:
        self._connecting = True
        self._connect_from_port = port
        self._preview_line = ConnectionPreviewItem()
        self.addItem(self._preview_line)
        self._preview_line.update_path(port.center_scene_pos, scene_pos)

    def update_connection_drag(self, scene_pos: QPointF) -> None:
        if self._preview_line and self._connect_from_port:
            self._preview_line.update_path(self._connect_from_port.center_scene_pos, scene_pos)

    def end_connection_drag(self, target_port: Optional[PortItem]) -> None:
        if self._preview_line:
            self.removeItem(self._preview_line)
            self._preview_line = None

        # Clear port highlight
        if self._highlighted_port:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None

        if not self._connecting or not self._connect_from_port:
            self._connecting = False
            return

        from_port = self._connect_from_port
        self._connecting = False
        self._connect_from_port = None

        if not target_port:
            return
        if from_port.node_id == target_port.node_id:
            return
        if from_port.port_type == target_port.port_type:
            return

        # Determine direction
        if from_port.port_type == "output":
            from_node, from_output = from_port.node_id, from_port.port_name
            to_node, to_input = target_port.node_id, target_port.port_name
        else:
            from_node, from_output = target_port.node_id, target_port.port_name
            to_node, to_input = from_port.node_id, from_port.port_name

        try:
            self.bridge.create_connection(from_node, from_output, to_node, to_input)
            # Rebuild connections from registry
        except Exception as e:
            print(f"Connection failed: {e}")
        
        self._rebuild_connections()

    def cancel_connection_drag(self) -> None:
        if self._preview_line:
            self.removeItem(self._preview_line)
            self._preview_line = None
        # Clear port highlight
        if self._highlighted_port:
            self._highlighted_port.set_highlighted(False)
            self._highlighted_port = None
        self._connecting = False
        self._connect_from_port = None

    @property
    def is_connecting(self) -> bool:
        return self._connecting

    def _rebuild_connections(self) -> None:
        """Remove all connection items and rebuild from registry."""
        for item in self._connection_items:
            self.removeItem(item)
        self._connection_items.clear()

        from minicortex.core.registry import get_connections
        for conn in get_connections():
            self.add_connection_item(conn)

    # ── Selection ────────────────────────────────────────────────────────

    def select_connection(self, item: ConnectionItem) -> None:
        if self._selected_connection:
            self._selected_connection.set_selected(False)
        self._selected_connection = item
        item.set_selected(True)
        self.connection_selected.emit(item.conn_data)

    def deselect_all(self) -> None:
        if self._selected_connection:
            self._selected_connection.set_selected(False)
            self._selected_connection = None
        self.clearSelection()
        self.selection_cleared.emit()

    def clear_connection_selection(self) -> None:
        if self._selected_connection:
            self._selected_connection.set_selected(False)
            self._selected_connection = None

    def delete_selected(self) -> None:
        """Delete selected node or connection."""
        if self._selected_connection:
            d = self._selected_connection.conn_data
            try:
                self.bridge.delete_connection(
                    d["from_node"], d["from_output"], d["to_node"], d["to_input"]
                )
            except ValueError:
                pass
            self.remove_connection_item(d)
            self._selected_connection = None
            return

        selected = [item for item in self.selectedItems() if isinstance(item, NodeItem)]
        for item in selected:
            try:
                self.bridge.delete_node(item.node_id)
            except ValueError:
                pass
            self.remove_node_item(item.node_id)
            self._rebuild_connections()

    # ── Signal handlers ──────────────────────────────────────────────────

    def _on_node_moved(self, node_id: str, x: float, y: float) -> None:
        self.bridge.update_node_position(node_id, x, y)

    def _on_property_changed(self, node_id: str, prop_key: str, value: object) -> None:
        try:
            self.bridge.set_property(node_id, prop_key, value)
        except Exception as e:
            print(f"Property update failed: {e}")

    def _on_action_triggered(self, node_id: str, action_key: str) -> None:
        try:
            self.bridge.execute_action(node_id, action_key)
        except Exception as e:
            print(f"Action failed: {e}")

    def _on_reload_requested(self, node_id: str) -> None:
        try:
            new_node = self.bridge.reload_node(node_id)
            # Remove old item and rebuild
            self.remove_node_item(node_id)
            self.add_node_item(new_node.get_schema())
            self._rebuild_connections()
        except Exception as e:
            print(f"Reload failed: {e}")

    # ── Mouse events for connection creation ─────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # Right button → delete node or connection
        if event.button() == Qt.MouseButton.RightButton:
            items = self.items(event.scenePos())
            for item in items:
                if isinstance(item, NodeItem):
                    # Select and delete the node
                    self.clearSelection()
                    item.setSelected(True)
                    self.delete_selected()
                    return
                if isinstance(item, ConnectionItem):
                    self.select_connection(item)
                    self.delete_selected()
                    return
        if event.button() == Qt.MouseButton.LeftButton:
            # Use items() to find items at the click position
            items = self.items(event.scenePos())
            
            # Find the topmost node at this position (if any)
            top_node = None
            for item in items:
                if isinstance(item, NodeItem):
                    top_node = item
                    break
            
            # Check for port directly under cursor
            clicked_port = None
            for item in items:
                if isinstance(item, PortItem):
                    clicked_port = item
                    break
            
            # If no exact port under cursor, try label hit
            if clicked_port is None:
                clicked_port = self._port_at_label(event.scenePos())
            
            # If a port was found, only start connection drag if:
            # 1. There's no node at this position, OR
            # 2. The port belongs to the topmost node at this position
            if clicked_port:
                if top_node is None:
                    # No node blocking - start connection drag
                    self.start_connection_drag(clicked_port, event.scenePos())
                    return
                elif top_node.node_id == clicked_port.node_id:
                    # Port belongs to the topmost node - start connection drag
                    self.start_connection_drag(clicked_port, event.scenePos())
                    return
                # else: Port belongs to a different node behind the top node - ignore it
            
            # There's a node at this position (and no valid port to drag from)
            if top_node:
                # Check for connection item selection
                for item in items:
                    if isinstance(item, ConnectionItem):
                        self.deselect_all()
                        self.select_connection(item)
                        return
                # Clear any selected connection and let the node handle the event
                self.clear_connection_selection()
                super().mousePressEvent(event)
                return
            
            # No node under click - try proximity-based port snapping
            clicked_port = self._nearest_port_at(event.scenePos())
            if clicked_port:
                self.start_connection_drag(clicked_port, event.scenePos())
                return
            
            # Check for connection item
            for item in items:
                if isinstance(item, ConnectionItem):
                    self.deselect_all()
                    self.select_connection(item)
                    return
            # No port or connection clicked; clear any selected connection
            self.clear_connection_selection()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._connecting:
            self.update_connection_drag(event.scenePos())
            self._update_port_highlight(event.scenePos())
            return
        # Also update port highlight when not connecting (for label hover)
        self._update_port_highlight(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._connecting:
            # Use items() which is the scene's method for finding items at a position
            items = self.items(event.scenePos())
            target = None
            for item in items:
                if isinstance(item, PortItem):
                    target = item
                    break
            # If no exact port under cursor, try label hit
            if target is None:
                target = self._port_at_label(event.scenePos())
            # If no label hit, try proximity-based snapping
            if target is None:
                target = self._nearest_port_at(event.scenePos())
            self.end_connection_drag(target)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        # Allow keys to be handled by focused widget (e.g., property input)
        focus = self.focusItem()
        if focus:
            # Check if focus is a proxy widget or a child of a proxy widget
            from PySide6.QtWidgets import QGraphicsProxyWidget
            item = focus
            while item:
                if isinstance(item, QGraphicsProxyWidget):
                    super().keyPressEvent(event)
                    return
                item = item.parentItem()

        # Shift+D → duplicate selected node
        if event.key() == Qt.Key.Key_D and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self.duplicate_selected()
            return

        super().keyPressEvent(event)

    def duplicate_selected(self) -> None:
        """Duplicate selected nodes (Shift+D)."""
        selected = [item for item in self.selectedItems() if isinstance(item, NodeItem)]
        if not selected:
            return

        for item in selected:
            node_type = item.node_type
            # Offset the new node by 30 pixels
            x = item.x() + 60
            y = item.y() + 30
            try:
                new_node = self.bridge.create_node(node_type, x, y)
                new_item = self.add_node_item(new_node.get_schema())
                # Copy properties from original to new node
                self._copy_node_properties(item, new_item)
            except Exception as e:
                print(f"Failed to duplicate node: {e}")

    def _copy_node_properties(self, source_item: NodeItem, target_item: NodeItem) -> None:
        """Copy property values from source node to target node."""
        for prop in source_item.schema.get("properties", []):
            key = prop.get("key")
            if key and prop.get("value") is not None:
                try:
                    self.bridge.set_property(target_item.node_id, key, prop["value"])
                except Exception as e:
                    print(f"Failed to copy property {key}: {e}")
