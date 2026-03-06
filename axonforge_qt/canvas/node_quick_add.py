"""
NodeQuickAddPopup — Shift+A quick node picker with search + cascading categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

from PySide6.QtCore import QEvent, QPoint, QRect, QTimer, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


ROLE_KIND = Qt.ItemDataRole.UserRole + 1
ROLE_NAME = Qt.ItemDataRole.UserRole + 2
ROLE_NODE_TYPE = Qt.ItemDataRole.UserRole + 3
ROLE_PATH = Qt.ItemDataRole.UserRole + 4

KIND_CATEGORY = "category"
KIND_NODE = "node"


@dataclass
class _BranchNode:
    children: Dict[str, "_BranchNode"] = field(default_factory=dict)
    node_types: List[str] = field(default_factory=list)


class _CascadeCategoryPopup(QWidget):
    """One level of cascading category/node list."""

    node_selected = Signal(str)

    def __init__(
        self,
        branch_node: _BranchNode,
        path_parts: List[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.ToolTip
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.X11BypassWindowManagerHint,
        )
        self._branch_node = branch_node
        self._path_parts = path_parts
        self._child_popup: Optional[_CascadeCategoryPopup] = None

        self.setObjectName("node_quick_add_cascade")
        self.setMinimumWidth(280)
        self.setMaximumHeight(420)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel(" > ".join(path_parts) if path_parts else "Categories")
        title.setStyleSheet("color: #8fa4ff; font-size: 11px;")
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setMouseTracking(True)
        self._list.viewport().setMouseTracking(True)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setAlternatingRowColors(False)
        self._list.setStyleSheet(
            "QListWidget { background: #0f1526; color: #e6f1ff; border: 1px solid #26324d; }"
            "QListWidget::item { padding: 4px 6px; }"
            "QListWidget::item:selected { background: #1e2d50; color: #00f5ff; }"
            "QListWidget::item:hover { background: #172240; }"
        )
        layout.addWidget(self._list)

        self._populate()
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        self._list.itemEntered.connect(self._on_item_entered)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemActivated.connect(self._on_item_clicked)

    def _populate(self) -> None:
        self._list.clear()
        for category_name in sorted(self._branch_node.children.keys()):
            item = QListWidgetItem(f"{category_name}  ▸")
            item.setData(ROLE_KIND, KIND_CATEGORY)
            item.setData(ROLE_NAME, category_name)
            item.setData(ROLE_PATH, self._path_parts + [category_name])
            self._list.addItem(item)

        for node_type in sorted(self._branch_node.node_types):
            item = QListWidgetItem(node_type)
            item.setData(ROLE_KIND, KIND_NODE)
            item.setData(ROLE_NODE_TYPE, node_type)
            item.setData(ROLE_PATH, self._path_parts + [node_type])
            self._list.addItem(item)

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _on_item_entered(self, item: QListWidgetItem) -> None:
        self._list.setCurrentItem(item)
        kind = item.data(ROLE_KIND)
        if kind != KIND_CATEGORY:
            self._close_child_popup()
            return

        category_name = str(item.data(ROLE_NAME))
        child_node = self._branch_node.children.get(category_name)
        if child_node is None:
            self._close_child_popup()
            return

        item_rect = self._list.visualItemRect(item)
        anchor_global = self._list.viewport().mapToGlobal(item_rect.topRight())
        self._open_child_popup(child_node, self._path_parts + [category_name], anchor_global)

    def _on_current_item_changed(
        self,
        current: Optional[QListWidgetItem],
        previous: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            self._close_child_popup()
            return

        kind = current.data(ROLE_KIND)
        if kind != KIND_CATEGORY:
            self._close_child_popup()
            return

    def _open_child_popup(self, node: _BranchNode, path_parts: List[str], anchor_global: QPoint) -> None:
        if self._child_popup is not None:
            # Keep the same popup if path matches; otherwise replace.
            if self._child_popup._path_parts == path_parts:
                return
            self._child_popup.close()
            self._child_popup = None

        child = _CascadeCategoryPopup(node, path_parts, parent=self)
        child.node_selected.connect(self.node_selected.emit)
        x = max(anchor_global.x() + 6, self.frameGeometry().right() + 6)
        child.move(QPoint(x, anchor_global.y()))
        child.show()
        self._child_popup = child

    def _close_child_popup(self) -> None:
        if self._child_popup is not None:
            self._child_popup.close()
            self._child_popup = None

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if item.data(ROLE_KIND) == KIND_NODE:
            node_type = str(item.data(ROLE_NODE_TYPE))
            self.node_selected.emit(node_type)

    def closeEvent(self, event) -> None:
        self._close_child_popup()
        super().closeEvent(event)


class NodeQuickAddPopup(QWidget):
    """Quick-add popup with searchable nodes and cascading category browser."""

    node_selected = Signal(str)

    def __init__(
        self,
        node_palette: Dict[str, List[Type]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.X11BypassWindowManagerHint,
        )
        self.setObjectName("node_quick_add_popup")
        self.setMinimumWidth(420)
        self.setMaximumHeight(520)
        self.setWindowTitle("Add Node")

        self._root_branch = self._build_branch_tree(node_palette)
        self._flat_nodes = self._build_flat_nodes(node_palette)
        self._root_cascade: Optional[_CascadeCategoryPopup] = None
        self._event_filter_installed = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Add Node")
        label.setStyleSheet("color: #8fa4ff; font-size: 12px; font-weight: 600;")
        row.addWidget(label)
        row.addStretch()
        hint = QLabel("Shift+A")
        hint.setStyleSheet("color: #6f87b9; font-size: 10px;")
        row.addWidget(hint)
        root_layout.addLayout(row)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search nodes... (Enter adds first)")
        self._search.setStyleSheet(
            "QLineEdit { background: #0f1526; color: #e6f1ff; border: 1px solid #26324d; padding: 4px 6px; }"
            "QLineEdit:focus { border-color: #00f5ff; }"
        )
        root_layout.addWidget(self._search)

        self._list = QListWidget()
        self._list.setMouseTracking(True)
        self._list.viewport().setMouseTracking(True)
        self._list.setStyleSheet(
            "QListWidget { background: #0f1526; color: #e6f1ff; border: 1px solid #26324d; }"
            "QListWidget::item { padding: 4px 6px; }"
            "QListWidget::item:selected { background: #1e2d50; color: #00f5ff; }"
            "QListWidget::item:hover { background: #172240; }"
        )
        root_layout.addWidget(self._list, 1)

        self.setStyleSheet(
            "QWidget#node_quick_add_popup { background: #11172a; border: 1px solid #26324d; }"
            "QWidget#node_quick_add_cascade { background: #11172a; border: 1px solid #26324d; }"
        )

        self._search.textChanged.connect(self._on_search_changed)
        self._search.returnPressed.connect(self._activate_first_result)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemActivated.connect(self._on_item_clicked)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        self._list.itemEntered.connect(self._on_item_entered)

        self._populate_browse_root()

    def focus_search(self) -> None:
        if not self._event_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
                self._event_filter_installed = True
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(0, lambda: self._search.setFocus(Qt.FocusReason.ActiveWindowFocusReason))
        QTimer.singleShot(0, self._search.selectAll)

    def _build_branch_tree(self, node_palette: Dict[str, List[Type]]) -> _BranchNode:
        root = _BranchNode()
        for branch_path, class_list in node_palette.items():
            current = root
            parts = [p for p in branch_path.split("/") if p]
            for part in parts:
                current = current.children.setdefault(part, _BranchNode())
            for cls in class_list:
                node_type = cls.__name__
                if node_type not in current.node_types:
                    current.node_types.append(node_type)
        return root

    def _build_flat_nodes(self, node_palette: Dict[str, List[Type]]) -> List[Tuple[str, str]]:
        flat: List[Tuple[str, str]] = []
        for branch_path, class_list in node_palette.items():
            parts = [p for p in branch_path.split("/") if p]
            for cls in class_list:
                node_type = cls.__name__
                full_path = " > ".join(parts + [node_type]) if parts else node_type
                flat.append((full_path, node_type))
        flat.sort(key=lambda t: t[0].lower())
        return flat

    def _populate_browse_root(self) -> None:
        self._list.clear()
        for category_name in sorted(self._root_branch.children.keys()):
            item = QListWidgetItem(f"{category_name}  ▸")
            item.setData(ROLE_KIND, KIND_CATEGORY)
            item.setData(ROLE_NAME, category_name)
            self._list.addItem(item)

        for node_type in sorted(self._root_branch.node_types):
            item = QListWidgetItem(node_type)
            item.setData(ROLE_KIND, KIND_NODE)
            item.setData(ROLE_NODE_TYPE, node_type)
            self._list.addItem(item)

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _on_search_changed(self, query: str) -> None:
        query = query.strip()
        self._close_root_cascade()
        if not query:
            self._populate_browse_root()
            return

        ranked = self._search_nodes(query)
        self._list.clear()
        for full_path, node_type, _score in ranked[:300]:
            item = QListWidgetItem(full_path)
            item.setData(ROLE_KIND, KIND_NODE)
            item.setData(ROLE_NODE_TYPE, node_type)
            self._list.addItem(item)
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    @staticmethod
    def _subsequence_score(needle: str, haystack: str) -> int:
        if not needle:
            return 0
        pos = 0
        gap_penalty = 0
        for char in needle:
            idx = haystack.find(char, pos)
            if idx < 0:
                return -1
            gap_penalty += idx - pos
            pos = idx + 1
        return max(1, 400 - gap_penalty)

    def _score_match(self, query: str, full_path: str, node_type: str) -> int:
        qn = self._normalize_text(query)
        if not qn:
            return 0
        nn = self._normalize_text(node_type)
        pn = self._normalize_text(full_path)

        if nn.startswith(qn):
            return 1000 - len(nn)
        idx = nn.find(qn)
        if idx >= 0:
            return 850 - idx
        seq = self._subsequence_score(qn, nn)
        if seq > 0:
            return seq

        if pn.startswith(qn):
            return 650 - len(pn)
        pidx = pn.find(qn)
        if pidx >= 0:
            return 550 - pidx
        pseq = self._subsequence_score(qn, pn)
        if pseq > 0:
            return pseq // 2
        return -1

    def _search_nodes(self, query: str) -> List[Tuple[str, str, int]]:
        matches: List[Tuple[str, str, int]] = []
        for full_path, node_type in self._flat_nodes:
            score = self._score_match(query, full_path, node_type)
            if score >= 0:
                matches.append((full_path, node_type, score))
        matches.sort(key=lambda x: (-x[2], len(x[1]), x[0].lower()))
        return matches

    def _on_item_entered(self, item: QListWidgetItem) -> None:
        self._list.setCurrentItem(item)
        if self._search.text().strip():
            self._close_root_cascade()
            return
        kind = item.data(ROLE_KIND)
        if kind != KIND_CATEGORY:
            self._close_root_cascade()
            return

        category_name = str(item.data(ROLE_NAME))
        child_node = self._root_branch.children.get(category_name)
        if child_node is None:
            self._close_root_cascade()
            return

        rect = self._list.visualItemRect(item)
        anchor_global = self._list.viewport().mapToGlobal(rect.topRight())
        self._open_root_cascade(child_node, [category_name], anchor_global)

    def _on_current_item_changed(
        self,
        current: Optional[QListWidgetItem],
        previous: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            self._close_root_cascade()
            return
        if self._search.text().strip():
            self._close_root_cascade()
            return

        kind = current.data(ROLE_KIND)
        if kind != KIND_CATEGORY:
            self._close_root_cascade()
            return

    def _open_root_cascade(
        self,
        branch_node: _BranchNode,
        path_parts: List[str],
        anchor_global: QPoint,
    ) -> None:
        if self._root_cascade is not None:
            if self._root_cascade._path_parts == path_parts:
                return
            self._root_cascade.close()
            self._root_cascade = None

        popup = _CascadeCategoryPopup(branch_node, path_parts, parent=self)
        popup.node_selected.connect(self._on_node_selected)
        x = max(anchor_global.x() + 6, self.frameGeometry().right() + 6)
        popup.move(QPoint(x, anchor_global.y()))
        popup.show()
        self._root_cascade = popup

    def _close_root_cascade(self) -> None:
        if self._root_cascade is not None:
            self._root_cascade.close()
            self._root_cascade = None

    def _collect_popup_geometries(self) -> List[QRect]:
        geoms = [self.frameGeometry()]
        popup = self._root_cascade
        while popup is not None:
            geoms.append(popup.frameGeometry())
            popup = popup._child_popup
        return geoms

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if item.data(ROLE_KIND) != KIND_NODE:
            return
        node_type = str(item.data(ROLE_NODE_TYPE))
        self._on_node_selected(node_type)

    def _on_node_selected(self, node_type: str) -> None:
        self.node_selected.emit(node_type)
        self.close()

    def _activate_first_result(self) -> None:
        item = self._list.currentItem()
        if item is None and self._list.count() > 0:
            item = self._list.item(0)
        if item is not None and item.data(ROLE_KIND) == KIND_NODE:
            self._on_node_selected(str(item.data(ROLE_NODE_TYPE)))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            item = self._list.currentItem()
            if item is None and self._list.count() > 0:
                item = self._list.item(0)
            if item is not None and item.data(ROLE_KIND) == KIND_NODE:
                self._on_node_selected(str(item.data(ROLE_NODE_TYPE)))
                return
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if not self.isVisible():
            return False

        if event.type() == QEvent.Type.MouseButtonPress:
            global_pos = event.globalPosition().toPoint()
            if not any(geom.contains(global_pos) for geom in self._collect_popup_geometries()):
                self.close()
                return False
        return False

    def closeEvent(self, event) -> None:
        if self._event_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._event_filter_installed = False
        self._close_root_cascade()
        super().closeEvent(event)
