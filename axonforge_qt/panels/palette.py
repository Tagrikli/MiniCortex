"""
PalettePanel — Left-side drawer showing available node types grouped by branch path.

Supports drag-and-drop onto the canvas to create nodes.
Branch paths like "Input/Noise" create nested tree structure.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Type, TYPE_CHECKING, Union

from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QDrag, QColor, QFont, QPainter, QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QAbstractItemView
)

if TYPE_CHECKING:
    from ..bridge import BridgeAPI

BRANCH_COLOR_ROLE = Qt.ItemDataRole.UserRole + 40
BRANCH_MARKER_SIZE = 8


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


def _dim_color(hex_color: str, saturation_scale: float = 0.72, value_scale: float = 0.88) -> str:
    """Return a dimmer variant of the given color, preserving hue."""
    base = QColor(hex_color)
    h, s, v, a = base.getHsv()
    # QColor may return achromatic colors with hue -1; keep them stable.
    if h < 0:
        h = 0
    s = max(35, min(255, int(s * saturation_scale)))
    v = max(75, min(255, int(v * value_scale)))
    return QColor.fromHsv(h, s, v, a).name()


def _create_square_pixmap(color: str = "#C0C0C0", size: int = BRANCH_MARKER_SIZE) -> QPixmap:
    """Create a square marker pixmap for branch indicators."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    square = QColor(color)
    painter.setBrush(square)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.end()

    return pixmap


_square_cache: dict[str, QPixmap] = {}


def _get_square_pixmap(color: str) -> QPixmap:
    """Get or create a square marker pixmap for the given color."""
    cached = _square_cache.get(color)
    if cached is not None:
        return cached
    pixmap = _create_square_pixmap(color=color)
    _square_cache[color] = pixmap
    return pixmap


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
            # Marker stays fixed, but keep icon synced after state toggles.
            self._update_branch_icon(item)

    def _update_branch_icon(self, item: QTreeWidgetItem) -> None:
        """Use a fixed square marker icon for branch items."""
        marker_color = item.data(0, BRANCH_COLOR_ROLE) or "#8fa4ff"
        icon = QIcon(_get_square_pixmap(color=marker_color))
        item.setIcon(0, icon)

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        """Handle item expansion - keep marker icon in sync."""
        self._update_branch_icon(item)

    def _on_item_collapsed(self, item: QTreeWidgetItem) -> None:
        """Handle item collapse - keep marker icon in sync."""
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
                mime.setData("application/x-axonforge-node-type", node_type.encode())
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
    
    def _build_tree_items(
        self,
        tree_or_parent: Union[QTreeWidget, QTreeWidgetItem],
        tree_dict: Dict,
        parent_path: List[str],
        parent_branch_color: str | None = None,
        root_category_color: str | None = None,
    ) -> None:
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

            if parent_path:
                # Subcategories progressively dim from their parent branch color.
                branch_color = _dim_color(parent_branch_color or "#8fa4ff")
                main_category_color = root_category_color or parent_branch_color or branch_color
            else:
                # Top-level category owns the main color.
                main_category_color = _generate_neon_color(name)
                branch_color = main_category_color
            
            # Check if this branch has children (either nodes or sub-branches)
            has_children = (isinstance(value, dict) and (
                "_nodes" in value or 
                any(k != "_nodes" for k in value.keys())
            ))
            if has_children:
                branch_item.setData(0, BRANCH_COLOR_ROLE, branch_color)
                icon = QIcon(_get_square_pixmap(color=branch_color))
                branch_item.setIcon(0, icon)
            
            # Keep branch text neutral; color is represented by the square marker.
            branch_item.setForeground(0, QColor("#cfd8ee"))
            
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
                self._build_tree_items(
                    branch_item,
                    value,
                    current_path,
                    parent_branch_color=branch_color,
                    root_category_color=main_category_color,
                )
            
            # Start collapsed by default.
            branch_item.setExpanded(False)

    def _on_rediscover(self) -> None:
        self.bridge.rediscover()
        self._build_palette()
