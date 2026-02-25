"""
MiniCortex Server - Main FastAPI application.

This module provides the main FastAPI application and server initialization.
"""

from typing import Dict, List, Optional, Any, Type

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ..core.node import Node
from ..core.registry import clear_node_registry, register_node, get_connections
from .state import state
from .models import ViewportState
from .lifecycle import lifespan
from .websocket import websocket_endpoint
from .routes.network import router as network_router, get_network_state
from .routes.nodes import router as nodes_router, build_topology_snapshot
from .routes.connections import router as connections_router
from .routes.workspaces import router as workspaces_router


# Create FastAPI application
app = FastAPI(title="MiniCortex Node Editor", lifespan=lifespan)

# Include routers
app.include_router(network_router)
app.include_router(nodes_router)
app.include_router(connections_router)
app.include_router(workspaces_router)


# ─────────────────────────────────────────────────────────────────────────────
# Config Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    """Get full application configuration."""
    return {
        "nodes": [node.get_schema() for node in state.nodes],
        "connections": get_connections(),
        "viewport": state.editor_viewport,
        "network": get_network_state(),
        "palette": state.node_palette,
    }


@app.get("/api/palette")
async def get_palette():
    """Get node palette for UI."""
    return state.node_palette


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_endpoint(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# Static Files
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Server Initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_server(
    nodes: List[Node],
    network=None,
    node_classes: Optional[Dict[str, Type[Node]]] = None,
    ws_fps: float = 60.0,
    node_palette: Optional[Dict[str, List[Type[Node]]]] = None,
):
    """
    Initialize the server with nodes, network, and node palette.
    
    Args:
        nodes: List of Node instances to register at startup
        network: Network manager instance
        node_classes: Legacy mapping of type name to Node class
        ws_fps: WebSocket broadcast frequency
        node_palette: Mapping of category name to list of Node classes
            {
                "Category Name": [NodeSubclass1, NodeSubclass2, ...],
                ...
            }
    """
    clear_node_registry()
    state.nodes = []
    state.network = network
    state.ws_fps = float(ws_fps)

    if node_palette is not None:
        # Build node_classes and node_palette from class-based palette
        state.node_classes = {}
        panels: List[Dict[str, Any]] = []
        for category, class_list in node_palette.items():
            panel_nodes: List[Dict[str, str]] = []
            for cls in class_list:
                type_name = cls.__name__
                panel_nodes.append({"type": type_name, "name": type_name})
                state.node_classes[type_name] = cls
            panels.append({"name": category, "nodes": panel_nodes})
        state.node_palette = {"panels": panels}
    else:
        # Fallback to explicit mapping (legacy behavior)
        state.node_classes = node_classes or {}

    state.editor_viewport = {"pan": {"x": 0.0, "y": 0.0}, "zoom": 1.0}
    
    for node in nodes:
        node_id = register_node(node)
        node.validate_required_methods()
        node.init()
        state.nodes.append(node)
        print(f"Registered node: {node_id}")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
