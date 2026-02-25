"""
Workspace API routes for saving and loading editor states.
"""
import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import state
from ...core.registry import (
    get_connections, clear_node_registry,
    register_node, add_connection
)
from ...core.node import Node

# Workspaces directory
WORKSPACES_DIR = Path("workspaces")
WORKSPACES_DIR.mkdir(exist_ok=True)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


class WorkspaceSave(BaseModel):
    name: str
    viewport: dict = {"pan": {"x": 0, "y": 0}, "zoom": 1.0}


class WorkspaceInfo(BaseModel):
    name: str
    created: Optional[float] = None
    modified: Optional[float] = None


def _get_workspace_path(name: str) -> Path:
    """Get the path for a workspace file."""
    # Sanitize name to prevent path traversal
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
    return WORKSPACES_DIR / f"{safe_name}.json"


def _serialize_workspace() -> dict:
    """Serialize current editor state to a workspace dict."""
    connections = get_connections()
    
    return {
        "version": 1,
        "viewport": state.editor_viewport,
        "nodes": [node.to_dict() for node in state.nodes],  # Use state.nodes list
        "connections": connections,
    }


def _deserialize_workspace(data: dict) -> dict:
    """Deserialize workspace data and restore editor state."""
    from ...core.registry import get_node_class
    
    # Clear current state
    clear_node_registry()
    state.nodes.clear()  # Also clear the state.nodes list
    
    # Restore nodes
    node_id_map = {}  # old_id -> new_id
    for node_data in data.get("nodes", []):
        node_type = node_data.get("type")
        node_class = get_node_class(node_type)
        if not node_class:
            continue
        
        node = node_class.from_dict(node_data)
        new_id = register_node(node)
        node_id_map[node_data.get("id")] = new_id
        node.init()
        state.nodes.append(node)  # Add to state.nodes list
    
    # Restore connections (with updated IDs)
    for conn in data.get("connections", []):
        from_node = node_id_map.get(conn["from_node"])
        to_node = node_id_map.get(conn["to_node"])
        if from_node and to_node:
            add_connection(
                from_node, conn["from_output"],
                to_node, conn["to_input"]
            )
    
    return {
        "viewport": data.get("viewport", {"pan": {"x": 0, "y": 0}, "zoom": 1.0}),
        "nodes_loaded": len(node_id_map),
        "connections_loaded": len(data.get("connections", [])),
    }


@router.get("")
async def list_workspaces() -> List[WorkspaceInfo]:
    """List all available workspaces."""
    workspaces = []
    for path in WORKSPACES_DIR.glob("*.json"):
        stat = path.stat()
        workspaces.append(WorkspaceInfo(
            name=path.stem,
            created=stat.st_ctime,
            modified=stat.st_mtime,
        ))
    return workspaces


@router.get("/current")
async def get_current_workspace():
    """Get the current workspace name."""
    return {"name": state.current_workspace}


@router.post("/save")
async def save_workspace(body: WorkspaceSave):
    """Save current editor state to a workspace file."""
    path = _get_workspace_path(body.name)
    
    workspace_data = _serialize_workspace()
    workspace_data["name"] = body.name
    
    with open(path, "w") as f:
        json.dump(workspace_data, f, indent=2)
    
    # Track current workspace
    state.current_workspace = body.name
    
    return {"status": "ok", "name": body.name, "path": str(path)}


@router.post("/load")
async def load_workspace(body: WorkspaceSave):
    """Load a workspace file and restore editor state."""
    from ..websocket import broadcast_state
    from .nodes import build_topology_snapshot
    
    path = _get_workspace_path(body.name)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    result = _deserialize_workspace(data)
    
    # Update viewport
    state.editor_viewport = result["viewport"]
    
    # Track current workspace
    state.current_workspace = body.name
    
    # Broadcast the new state to all connected clients
    await broadcast_state()
    
    return {
        "status": "ok",
        **result,
        "snapshot": build_topology_snapshot(),
    }


@router.delete("")
async def delete_workspace(body: WorkspaceSave):
    """Delete a workspace file."""
    path = _get_workspace_path(body.name)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    path.unlink()
    return {"status": "ok", "name": body.name}


@router.post("/clear")
async def clear_workspace():
    """Clear the current workspace (create empty new workspace)."""
    from ..websocket import broadcast_state
    from .nodes import build_topology_snapshot
    
    # Clear current state
    clear_node_registry()
    state.nodes.clear()
    
    # Clear current workspace tracking
    state.current_workspace = None
    
    # Reset viewport
    state.editor_viewport = {"pan": {"x": 0.0, "y": 0.0}, "zoom": 1.0}
    
    # Broadcast the cleared state
    await broadcast_state()
    
    return {
        "status": "ok",
        "snapshot": build_topology_snapshot(),
    }
