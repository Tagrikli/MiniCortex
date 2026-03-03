"""
BridgeAPI — Direct Python interface replacing the REST/WebSocket server layer.

All operations that the web UI performed via HTTP are now synchronous method calls.
The UI calls these directly; no serialization, no network overhead.
"""

import json
import importlib
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from minicortex.core.node import Node
from minicortex.core.registry import (
    register_node,
    unregister_node,
    get_node,
    get_all_nodes,
    get_connections,
    get_connections_for_node,
    add_connection,
    remove_connection,
    clear_node_registry,
)
from minicortex.core.descriptors.ports import _format_data_type
from minicortex.core.descriptors.node import (
    discover_nodes,
    rediscover_nodes,
    build_node_palette,
)
from minicortex.network.network import Network, NetworkError


WORKSPACES_DIR = Path("workspaces")
WORKSPACES_DIR.mkdir(exist_ok=True)


class BridgeAPI:
    """Synchronous API that the Qt UI calls directly."""

    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.network: Network = Network()
        self.node_classes: Dict[str, Type[Node]] = {}
        self.node_palette: Dict[str, List[Type[Node]]] = {}
        self.viewport: Dict[str, Any] = {"pan": {"x": 0.0, "y": 0.0}, "zoom": 1.0}
        self.current_workspace: Optional[str] = None

        # Shared display buffer: {node_id: {output_key: value}}
        self._display_buffer: Dict[str, Dict[str, Any]] = {}
        self._buffer_lock = threading.Lock()
        self._dirty = False

    # ── Initialisation ───────────────────────────────────────────────────

    def init(self) -> None:
        """Discover nodes and build palette."""
        discover_nodes()
        palette = build_node_palette()
        for category, class_list in palette.items():
            for cls in class_list:
                self.node_classes[cls.__name__] = cls
        self.node_palette = palette

    # ── Display buffer (shared with computation thread) ──────────────────

    def snapshot_displays(self) -> None:
        """Write current display outputs of every node into the shared buffer."""
        buf: Dict[str, Dict[str, Any]] = {}
        for node in self.nodes:
            outputs: Dict[str, Any] = {}
            for name, descriptor in getattr(node.__class__, "_outputs", {}).items():
                try:
                    val = getattr(node, name)
                except Exception:
                    val = None
                # Include display configuration from descriptor
                output_config = {}
                if hasattr(descriptor, 'color_mode'):
                    output_config['color_mode'] = descriptor.color_mode
                # Include scale configuration for bar/line charts
                if hasattr(descriptor, 'scale_mode'):
                    output_config['scale_mode'] = descriptor.scale_mode
                    output_config['scale_min'] = descriptor.scale_min
                    output_config['scale_max'] = descriptor.scale_max
                outputs[name] = {
                    'value': val,
                    'config': output_config
                }
            buf[node.node_id] = outputs
        with self._buffer_lock:
            self._display_buffer = buf
            self._dirty = True

    def read_display_buffer(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Read the display buffer if dirty. Returns None if clean."""
        with self._buffer_lock:
            if not self._dirty:
                return None
            self._dirty = False
            return self._display_buffer

    # ── Node CRUD ────────────────────────────────────────────────────────

    def create_node(self, node_type: str, x: float = 0, y: float = 0) -> Node:
        node_class = self.node_classes.get(node_type)
        if not node_class:
            raise ValueError(f"Unknown node type: {node_type}")
        node = node_class(x=x, y=y)
        register_node(node)
        node.validate_required_methods()
        try:
            node.init()
        except Exception as e:
            # If init fails, unregister the node and raise an error
            # so the node is not added to the workspace
            unregister_node(node.node_id)
            print(f"Node init failed for {node_type}: {e}")
            raise RuntimeError(f"Failed to initialize {node_type}: {e}") from e
        self.nodes.append(node)
        # Propagate current state so new node gets computed if it has inputs
        if self.network:
            try:
                self.network.propagate_current_state()
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Node creation error: {e}")
                if e.traceback:
                    print(e.traceback)
        self.snapshot_displays()
        return node

    def delete_node(self, node_id: str) -> None:
        if not get_node(node_id):
            raise ValueError(f"Node not found: {node_id}")
        for conn in get_connections_for_node(node_id):
            remove_connection(
                conn["from_node"], conn["from_output"],
                conn["to_node"], conn["to_input"],
            )
        self.nodes = [n for n in self.nodes if n.node_id != node_id]
        unregister_node(node_id)
        self.snapshot_displays()

    def get_node_schema(self, node_id: str) -> dict:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        return node.get_schema()

    def update_node_position(self, node_id: str, x: float, y: float) -> None:
        node = get_node(node_id)
        if node:
            node.position = {"x": float(x), "y": float(y)}

    # ── Properties & Actions ─────────────────────────────────────────────

    def set_property(self, node_id: str, prop_key: str, value: Any) -> None:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        setattr(node, prop_key, value)
        # Only propagate when network is not running (let running network handle processing)
        if self.network and not self.network.running:
            try:
                self.network.propagate_from_node(node_id, recompute_start=True)
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Property set error in '{node_id}.{prop_key}': {e}")
                if e.traceback:
                    print(e.traceback)
        self.snapshot_displays()

    def execute_action(self, node_id: str, action_key: str, params: Optional[dict] = None) -> Any:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        action = getattr(node, action_key)
        result = action(params)
        if self.network:
            try:
                self.network.propagate_from_node(node_id, recompute_start=True)
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Action execution error in '{node_id}.{action_key}': {e}")
                if e.traceback:
                    print(e.traceback)
        self.snapshot_displays()
        return result

    def set_output_enabled(self, node_id: str, output_key: str, enabled: bool) -> None:
        node = get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")
        if not hasattr(node, "_output_enabled") or node._output_enabled is None:
            node._output_enabled = {}
        node._output_enabled[output_key] = bool(enabled)

    # ── Connections ──────────────────────────────────────────────────────

    def create_connection(
        self, from_node: str, from_output: str, to_node: str, to_input: str
    ) -> None:
        fn = get_node(from_node)
        tn = get_node(to_node)
        if not fn or not tn:
            raise ValueError("Node not found")

        from_port = next(
            (p for p in getattr(fn.__class__, "_output_ports", {}).values()
             if p.name == from_output), None,
        )
        to_port = next(
            (p for p in getattr(tn.__class__, "_input_ports", {}).values()
             if p.name == to_input), None,
        )
        if not from_port or not to_port:
            raise ValueError("Port not found")

        if not self._is_type_compatible(from_port.data_type, to_port.data_type):
            raise ValueError(
                f"Incompatible types: {_format_data_type(from_port.data_type)} → "
                f"{_format_data_type(to_port.data_type)}"
            )

        # Remove existing connection to the same input
        for conn in get_connections():
            if conn["to_node"] == to_node and conn["to_input"] == to_input:
                remove_connection(
                    conn["from_node"], conn["from_output"],
                    conn["to_node"], conn["to_input"],
                )
                if self.network and hasattr(self.network, "_signals_now"):
                    key = (conn["from_node"], conn["from_output"])
                    self.network._signals_now.pop(key, None)

        if not add_connection(from_node, from_output, to_node, to_input):
            raise ValueError("Connection already exists")

        if self.network:
            try:
                self.network.propagate_from_node(to_node, recompute_start=True)
            except NetworkError as e:
                # Error is stored in network.last_error and node is tracked in failed_nodes
                print(f"Connection creation error from '{from_node}.{from_output}' to '{to_node}.{to_input}': {e}")
                if e.traceback:
                    print(e.traceback)
        self.snapshot_displays()

    def delete_connection(
        self, from_node: str, from_output: str, to_node: str, to_input: str
    ) -> None:
        if not remove_connection(from_node, from_output, to_node, to_input):
            raise ValueError("Connection not found")
        target = get_node(to_node)
        if target and hasattr(target, to_input):
            setattr(target, to_input, None)
        if self.network:
            key = (from_node, from_output)
            if hasattr(self.network, "_signals_now"):
                self.network._signals_now.pop(key, None)
        self.snapshot_displays()

    # ── Network Control ──────────────────────────────────────────────────

    def start_network(self) -> None:
        self.network.start()

    def stop_network(self) -> None:
        self.network.stop()

    def step_network(self) -> None:
        if self.network.running:
            self.network.stop()
        try:
            self.network.execute_step()
        except NetworkError as e:
            print(f"Network step error: {e}")
            if e.traceback:
                print(e.traceback)
            # error stored in network.last_error
        self.snapshot_displays()

    def set_network_speed(self, speed: float) -> None:
        self.network.speed = max(1.0, min(300.0, float(speed)))

    def get_network_state(self) -> dict:
        return {
            "running": bool(self.network.running),
            "speed": float(self.network.speed),
            "step": int(self.network.get_step_count()),
            "actual_hz": float(getattr(self.network, "actual_hz", 0.0)),
        }

    # ── Workspaces ───────────────────────────────────────────────────────

    def list_workspaces(self) -> List[dict]:
        result = []
        for path in WORKSPACES_DIR.glob("*.json"):
            stat = path.stat()
            result.append({
                "name": path.stem,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
            })
        return result

    def save_workspace(self, name: str) -> None:
        safe = "".join(c for c in name if c.isalnum() or c in "_-")
        path = WORKSPACES_DIR / f"{safe}.json"
        data = {
            "version": 1,
            "name": name,
            "viewport": self.viewport,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": get_connections(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.current_workspace = name

    def load_workspace(self, name: str) -> dict:
        safe = "".join(c for c in name if c.isalnum() or c in "_-")
        path = WORKSPACES_DIR / f"{safe}.json"
        if not path.exists():
            raise FileNotFoundError(f"Workspace not found: {name}")
        with open(path) as f:
            data = json.load(f)

        clear_node_registry()
        self.nodes.clear()

        node_id_map: Dict[str, str] = {}
        for nd in data.get("nodes", []):
            cls = self.node_classes.get(nd.get("type"))
            if not cls:
                continue
            node = cls.from_dict(nd)
            new_id = register_node(node)
            node_id_map[nd.get("id")] = new_id
            node.init()
            self.nodes.append(node)

        for conn in data.get("connections", []):
            fn = node_id_map.get(conn["from_node"])
            tn = node_id_map.get(conn["to_node"])
            if fn and tn:
                add_connection(fn, conn["from_output"], tn, conn["to_input"])

        self.viewport = data.get("viewport", {"pan": {"x": 0, "y": 0}, "zoom": 1.0})
        self.current_workspace = name
        self.snapshot_displays()
        return self.viewport

    def delete_workspace(self, name: str) -> None:
        safe = "".join(c for c in name if c.isalnum() or c in "_-")
        path = WORKSPACES_DIR / f"{safe}.json"
        if not path.exists():
            raise FileNotFoundError(f"Workspace not found: {name}")
        path.unlink()
        if self.current_workspace == name:
            self.current_workspace = None

    def clear_workspace(self) -> None:
        clear_node_registry()
        self.nodes.clear()
        self.current_workspace = None
        self.viewport = {"pan": {"x": 0.0, "y": 0.0}, "zoom": 1.0}
        self.snapshot_displays()

    # ── Hot Reload ───────────────────────────────────────────────────────

    def reload_node(self, node_id: str) -> Node:
        from minicortex.core.registry import _node_registry

        old = get_node(node_id)
        if not old:
            raise ValueError(f"Node not found: {node_id}")
        if not getattr(old.__class__, "dynamic", False):
            raise ValueError("Node is not dynamic")

        module_path = old.__class__.__module__
        module = sys.modules.get(module_path)
        if not module:
            raise RuntimeError(f"Module not found: {module_path}")

        importlib.reload(module)
        new_class = getattr(module, old.__class__.__name__)
        if not getattr(new_class, "dynamic", False):
            raise ValueError("Reloaded class is no longer dynamic")

        new_inputs = set(getattr(new_class, "_input_ports", {}).keys())
        new_outputs = set(getattr(new_class, "_output_ports", {}).keys())
        for conn in get_connections_for_node(node_id):
            invalid = False
            if conn["from_node"] == node_id and conn["from_output"] not in new_outputs:
                invalid = True
            if conn["to_node"] == node_id and conn["to_input"] not in new_inputs:
                invalid = True
            if invalid:
                remove_connection(
                    conn["from_node"], conn["from_output"],
                    conn["to_node"], conn["to_input"],
                )

        new_node = new_class(x=old.position.get("x", 0), y=old.position.get("y", 0))
        new_node.node_id = node_id
        new_node.name = old.name

        # Only copy property values, not store values
        for prop_name in getattr(old.__class__, "_properties", {}).keys():
            if hasattr(old, prop_name):
                try:
                    setattr(new_node, prop_name, getattr(old, prop_name))
                except Exception:
                    pass

        if hasattr(old, "_output_enabled"):
            new_node._output_enabled = old._output_enabled.copy()

        for i, n in enumerate(self.nodes):
            if n.node_id == node_id:
                self.nodes[i] = new_node
                break

        _node_registry[node_id] = new_node
        if new_class.__name__ in self.node_classes:
            self.node_classes[new_class.__name__] = new_class

        new_node.init()
        self.snapshot_displays()
        return new_node

    def rediscover(self) -> None:
        new_palette = rediscover_nodes()
        for category, classes in new_palette.items():
            for cls in classes:
                self.node_classes[cls.__name__] = cls
        self.node_palette = new_palette

    # ── Topology Snapshot ────────────────────────────────────────────────

    def get_topology(self) -> dict:
        return {
            "nodes": [n.get_schema() for n in self.nodes],
            "connections": get_connections(),
            "viewport": self.viewport,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_type_compatible(
        output_type: Optional[Any],
        input_type: Optional[Any]
    ) -> bool:
        """
        Check if an output port can connect to an input port.

        Args:
            output_type: None (any), single Type, or list of Types
            input_type: None (any), single Type, or list of Types

        Returns:
            True if connection is allowed

        Rules:
            - None means "any type" (always compatible)
            - List means "any of these types" (OR logic)
            - Single type uses subclass checking
        """
        def normalize(t: Optional[Any]) -> List[Any]:
            """Normalize to list for uniform handling. Empty list = any type."""
            if t is None:
                return []
            if isinstance(t, (list, tuple, set)):
                return list(t)
            return [t]

        out_types = normalize(output_type)
        in_types = normalize(input_type)

        # None (any type) - always compatible
        if not out_types or not in_types:
            return True

        # Check if ANY output type is compatible with ANY input type
        for out_t in out_types:
            for in_t in in_types:
                if BridgeAPI._check_single_type(out_t, in_t):
                    return True

        return False

    @staticmethod
    def _check_single_type(output_type: Any, input_type: Any) -> bool:
        """Check compatibility between two single types."""
        # Subclass check for type objects
        if isinstance(output_type, type) and isinstance(input_type, type):
            return issubclass(output_type, input_type)
        # String comparison as fallback
        return str(output_type) == str(input_type)
