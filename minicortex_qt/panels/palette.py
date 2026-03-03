"""
PalettePanel — Left-side drawer showing available node types grouped by branch path.

Supports drag-and-drop onto the canvas to create nodes.
Branch paths like "Input/Noise" create nested tree structure.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Type, TYPE_CHECKING, Union

from PySide6.QtCore import Qt, QMimeData, QPoint, QSize
from PySide6.QtGui import QDrag, QColor, QMouseEvent, QFont, QPainter, QPixmap, QPolygon, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem,
    QAbstractItemView
)

if TYPE_CHECKING:
    from ..bridge import BridgeAPI


def _generate_neon_color(text: str) -> str:
    """Generate a neon-style color from text using hashing."""
    # Hash the text to get a consistent number
    hash_bytes = hashlib.md5(text.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:4], 'big')
    
    # Neon color ranges - high saturation and brightness
    # Hue: 0-360 (full spectrum)
    hue = hash_int % 360
    
    # Saturation: 80-100% for neon look
    saturation = 80 + (hash_int >> 8) % 21
    
    # Brightness: 80-100% for neon look (ensuring colors are not dark)
    brightness = 80 + (hash_int >> 16) % 21
    
    # Convert HSV to RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation / 100, brightness / 100)
    
    # Convert to hex
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def rgba_to_qcolor(rgba_str: str) -> QColor:
    """Convert 'rgba(r,g,b,a)' string to QColor."""
    import re
    m = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)', rgba_str)
    if not m:
        return QColor(136, 136, 136, 51)  # fallback gray with alpha
    r, g, b, a = map(float, m.groups())
    # alpha is 0-1, QColor expects 0-255
    alpha = int(a * 255)
    return QColor(int(r), int(g), int(b), alpha)


def _create_triangle_pixmap(pointing_down: bool = False, color: str = "#C0C0C0", size: int = 12) -> QPixmap:
    """Create a triangle pixmap for branch indicators.
    
    Args:
        pointing_down: If True, creates a downward-pointing triangle (expanded),
                       otherwise creates a rightward-pointing triangle (collapsed).
        color: The color of the triangle (bright grayish by default).
        size: The size of the triangle in pixels.
    
    Returns:
        QPixmap containing the triangle.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    # Set the brush and pen for the triangle
    painter.setBrush(QColor(color))
    painter.setPen(QColor(color))
    
    # Create triangle points
    if pointing_down:
        # Downward-pointing triangle (expanded state)
        points = QPolygon([
            QPoint(2, 2),           # top-left
            QPoint(size - 2, 2),    # top-right
            QPoint(size // 2, size - 2)  # bottom-center
        ])
    else:
        # Rightward-pointing triangle (collapsed state)
        points = QPolygon([
            QPoint(2, 2),           # top-left
            QPoint(2, size - 2),    # bottom-left
            QPoint(size - 2, size // 2)  # right-center
        ])
    
    painter.drawPolygon(points)
    painter.end()
    
    return pixmap


# Pre-create the triangle pixmaps for use in stylesheet (lazy initialization)
_triangle_right_pixmap: QPixmap | None = None
_triangle_down_pixmap: QPixmap | None = None


def _get_triangle_right_pixmap() -> QPixmap:
    """Get or create the right-pointing triangle pixmap."""
    global _triangle_right_pixmap
    if _triangle_right_pixmap is None:
        _triangle_right_pixmap = _create_triangle_pixmap(pointing_down=False)
    return _triangle_right_pixmap


def _get_triangle_down_pixmap() -> QPixmap:
    """Get or create the down-pointing triangle pixmap."""
    global _triangle_down_pixmap
    if _triangle_down_pixmap is None:
        _triangle_down_pixmap = _create_triangle_pixmap(pointing_down=True)
    return _triangle_down_pixmap


class _PaletteTreeWidget(QTreeWidget):
    """Tree widget displaying node categories and types with custom drag."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setIndentation(20)
        self.setExpandsOnDoubleClick(False)  # Disable double-click expand, we'll handle single click
        self.itemClicked.connect(self._on_item_clicked)
        # Connect expanded/collapsed signals for icon updates
        self.itemExpanded.connect(self._on_item_expanded)
        self.itemCollapsed.connect(self._on_item_collapsed)
        self.setObjectName("palette_tree")

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle single click on items - toggle expand/collapse for branch items."""
        # Check if this is a branch item (not selectable, has children)
        is_branch = not (item.flags() & Qt.ItemFlag.ItemIsSelectable)
        if is_branch and item.childCount() > 0:
            item.setExpanded(not item.isExpanded())
            # Update the triangle icon based on new expanded state
            self._update_branch_icon(item)

    def _update_branch_icon(self, item: QTreeWidgetItem) -> None:
        """Update the triangle icon based on expanded state."""
        if item.isExpanded():
            # Expanded: show downward-pointing triangle
            icon = QIcon(_get_triangle_down_pixmap())
        else:
            # Collapsed: show rightward-pointing triangle
            icon = QIcon(_get_triangle_right_pixmap())
        item.setIcon(0, icon)

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        """Handle item expansion - update icon to downward triangle."""
        self._update_branch_icon(item)

    def _on_item_collapsed(self, item: QTreeWidgetItem) -> None:
        """Handle item collapse - update icon to rightward triangle."""
        self._update_branch_icon(item)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        # Only allow drag from leaf items (node types)
        # Leaf items have data stored in UserRole
        if item:
            node_type = item.data(0, Qt.ItemDataRole.UserRole)
            if node_type:
                drag = QDrag(self)
                mime = QMimeData()
                mime.setText(node_type)
                mime.setData("application/x-minicortex-node-type", node_type.encode())
                drag.setMimeData(mime)
                drag.exec(Qt.DropAction.CopyAction)
                # Clear the selection and current item to remove highlight and border
                self.clearSelection()
                self.setCurrentItem(None)
                self.repaint()
                return
        # Don't call super().startDrag() for branch items - they are not draggable
        super().startDrag(supportedActions)


class PalettePanel(QWidget):
    """Left-side node palette panel."""

    def __init__(self, bridge: BridgeAPI, parent=None) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self.setObjectName("palette_panel")
        # Allow resizing via QSplitter in main window

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("palette_header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)

        title = QLabel("NODES")
        title.setObjectName("palette_title")
        header_layout.addWidget(title)
        header_layout.addStretch()

        rediscover_btn = QPushButton("↻")
        rediscover_btn.setObjectName("rediscover_btn")
        rediscover_btn.setFixedSize(24, 24)
        rediscover_btn.clicked.connect(self._on_rediscover)
        header_layout.addWidget(rediscover_btn)

        layout.addWidget(header)

        # Tree widget
        self._tree = _PaletteTreeWidget()
        layout.addWidget(self._tree)

        self._build_palette()

    def _build_palette(self) -> None:
        """Build the tree from branch paths."""
        self._tree.clear()
        palette = self.bridge.node_palette
        
        # Build a tree structure from branch paths
        # tree_dict: nested dict where keys are branch names, values are either
        #            (nested dicts for sub-branches) or lists of node classes
        tree_dict: Dict = {}
        
        for branch_path, classes in palette.items():
            # Parse the branch path (e.g., "Input/Noise" -> ["Input", "Noise"])
            parts = branch_path.split("/")
            
            # Navigate/create the tree structure
            current = tree_dict
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {}
                if isinstance(current[part], list):
                    # This shouldn't happen with well-formed paths
                    current[part] = {}
                current = current[part]
            
            # At the leaf level, store the classes
            # Use a special key to store the node classes
            current["_nodes"] = classes
        
        # Now build the QTreeWidget from the tree_dict
        self._build_tree_items(self._tree, tree_dict, [])
    
    def _build_tree_items(self, tree_or_parent: Union[QTreeWidget, QTreeWidgetItem], tree_dict: Dict, parent_path: List[str]) -> None:
        """Recursively build tree items from the tree dictionary."""
        for name, value in sorted(tree_dict.items()):
            if name == "_nodes":
                # These are the actual node classes at this level
                continue
            
            # Create the branch item
            if isinstance(tree_or_parent, QTreeWidget):
                branch_item = QTreeWidgetItem([name])
                tree_or_parent.addTopLevelItem(branch_item)
            else:
                branch_item = QTreeWidgetItem([name])
                tree_or_parent.addChild(branch_item)
            
            # Branch items are not selectable and not draggable
            branch_item.setFlags(
                branch_item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsDragEnabled
            )
            
            # Set the triangle icon (right-pointing for collapsed, down-pointing for expanded)
            # Check if this branch has children (either nodes or sub-branches)
            has_children = (isinstance(value, dict) and (
                "_nodes" in value or 
                any(k != "_nodes" for k in value.keys())
            ))
            if has_children:
                # Default to expanded, so use down-pointing triangle
                icon = QIcon(_get_triangle_down_pixmap())
                branch_item.setIcon(0, icon)
            
            # Apply branch color using hash-based generation
            text_color = _generate_neon_color(name)
            branch_item.setForeground(0, QColor(text_color))
            
            # Set bold font for branch names
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            branch_item.setFont(0, font)
            
            # Build the full path for this branch
            current_path = parent_path + [name]
            
            # Add node classes (leaves) if any exist at this level
            if isinstance(value, dict) and "_nodes" in value:
                for cls in value["_nodes"]:
                    child = QTreeWidgetItem([cls.__name__])
                    child.setData(0, Qt.ItemDataRole.UserRole, cls.__name__)
                    # Apply subtle background styling
                    child.setBackground(0, QColor(136, 136, 136, 30))
                    branch_item.addChild(child)
            
            # Recursively add sub-branches
            if isinstance(value, dict):
                self._build_tree_items(branch_item, value, current_path)
            
            # Expand the branch by default
            branch_item.setExpanded(True)

    def _on_rediscover(self) -> None:
        self.bridge.rediscover()
        self._build_palette()
