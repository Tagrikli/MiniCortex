"""
Global state management for MiniCortex server.
"""

import asyncio
from typing import Dict, List, Optional, Any, Type

from ..core.node import Node


class ServerState:
    """Container for all server global state."""
    
    # Node instances
    nodes: List[Node] = []
    
    # Network manager
    network: Optional[Any] = None
    
    # WebSocket clients
    websocket_clients: set = set()
    
    # Background tasks
    computation_task: Optional[asyncio.Task] = None
    broadcast_task: Optional[asyncio.Task] = None
    
    # Editor state
    editor_viewport: Dict[str, Any] = {"pan": {"x": 0.0, "y": 0.0}, "zoom": 1.0}
    
    # WebSocket settings
    ws_fps: float = 60.0
    
    # Network timing
    network_last_step_time: Optional[float] = None
    network_actual_hz: float = 0.0
    network_max_hz: float = 300.0
    
    # Node palette (for UI)
    node_palette: Dict[str, Any] = {"panels": []}
    
    # Node class registry (type name -> class)
    node_classes: Dict[str, Type[Node]] = {}
    
    # Current workspace tracking
    current_workspace: Optional[str] = None


# Global state instance
state = ServerState()


def get_nodes() -> List[Node]:
    """Get all node instances."""
    return state.nodes


def get_network():
    """Get the network manager."""
    return state.network


def get_websocket_clients() -> set:
    """Get connected WebSocket clients."""
    return state.websocket_clients


def get_editor_viewport() -> Dict[str, Any]:
    """Get the current editor viewport state."""
    return state.editor_viewport


def get_node_classes() -> Dict[str, Type[Node]]:
    """Get the node class registry."""
    return state.node_classes


def get_node_palette() -> Dict[str, Any]:
    """Get the node palette for UI."""
    return state.node_palette


def get_ws_fps() -> float:
    """Get WebSocket broadcast FPS."""
    return state.ws_fps


def get_network_max_hz() -> float:
    """Get maximum network execution frequency."""
    return state.network_max_hz
