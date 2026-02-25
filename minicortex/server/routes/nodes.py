"""
Node CRUD API routes.
"""

import importlib
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..state import state
from ..models import (
    NodeCreate, PositionUpdate, PropertyUpdate, 
    ActionRequest, OutputEnableUpdate, TopologyContext
)
from ..websocket import broadcast_state
from ...core.registry import (
    get_node, register_node, unregister_node, get_connections_for_node
)


router = APIRouter(prefix="/api/nodes", tags=["nodes"])


def update_editor_viewport(viewport) -> None:
    """Update editor viewport state."""
    if viewport:
        state.editor_viewport = {
            "pan": {"x": float(viewport.pan.x), "y": float(viewport.pan.y)},
            "zoom": float(viewport.zoom),
        }


def build_topology_snapshot() -> dict:
    """Build a snapshot of the current topology."""
    from ...core.registry import get_connections
    return {
        "nodes": [node.get_schema() for node in state.nodes],
        "connections": get_connections(),
        "viewport": state.editor_viewport,
    }


@router.get("")
async def list_nodes():
    """List all nodes."""
    return [
        {
            "node_id": n.node_id,
            "node_type": n.node_type,
            "name": n.name,
            "position": n.position,
        }
        for n in state.nodes
    ]


@router.post("")
async def create_node(body: NodeCreate):
    """Create a new node."""
    node_class = state.node_classes.get(body.type)
    if not node_class:
        raise HTTPException(status_code=400, detail=f"Unknown node type: {body.type}")
    
    node = node_class(x=body.position.get("x", 0), y=body.position.get("y", 0))
    node_id = register_node(node)
    node.validate_required_methods()
    node.init()
    state.nodes.append(node)
    update_editor_viewport(body.viewport)
    await broadcast_state()
    
    return {
        "status": "ok",
        "node": node.get_schema(),
        "snapshot": build_topology_snapshot(),
    }


@router.get("/{node_id}")
async def get_node_schema(node_id: str):
    """Get a node's schema."""
    node = get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return node.get_schema()


@router.put("/{node_id}/position")
async def update_node_position(node_id: str, body: PositionUpdate):
    """Update a node's position."""
    node = get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    node.position = {"x": float(body.x), "y": float(body.y)}
    return {"status": "ok", "node_id": node_id, "position": node.position}


@router.delete("/{node_id}")
async def delete_node(node_id: str, body: Optional[TopologyContext] = None):
    """Delete a node."""
    if not get_node(node_id):
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    # Remove all connections involving this node
    for conn in get_connections_for_node(node_id):
        from ...core.registry import remove_connection
        remove_connection(
            conn["from_node"], conn["from_output"], 
            conn["to_node"], conn["to_input"]
        )
    
    # Remove node from list and registry
    state.nodes = [n for n in state.nodes if n.node_id != node_id]
    unregister_node(node_id)
    update_editor_viewport(body.viewport if body else None)
    await broadcast_state()
    
    return {"status": "ok", "node_id": node_id, "snapshot": build_topology_snapshot()}


@router.put("/{node_id}/properties/{property_name}")
async def set_node_property(node_id: str, property_name: str, body: PropertyUpdate):
    """Update a node property."""
    node = get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    try:
        setattr(node, property_name, body.value)
    except (KeyError, ValueError, AttributeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    await broadcast_state()
    
    return {
        "status": "ok",
        "node_id": node_id,
        "property": property_name,
        "value": body.value,
    }


@router.post("/{node_id}/actions/{action_name}")
async def execute_node_action(node_id: str, action_name: str, body: Optional[ActionRequest] = None):
    """Execute a node action."""
    node = get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    try:
        action = getattr(node, action_name)
        result = action(body.params if body else None)
    except (AttributeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    await broadcast_state()
    return {"status": "ok", "node_id": node_id, "action": action_name, "result": result}


@router.put("/{node_id}/outputs/{output_key}")
async def set_output_enabled(node_id: str, output_key: str, body: OutputEnableUpdate):
    """Enable/disable a visual output for a node."""
    node = get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    outputs = getattr(node.__class__, "_outputs", {})
    if output_key not in outputs:
        raise HTTPException(status_code=404, detail=f"Output not found: {output_key}")

    if not hasattr(node, "_output_enabled") or node._output_enabled is None:
        node._output_enabled = {}
    node._output_enabled[output_key] = bool(body.enabled)

    return {
        "status": "ok",
        "node_id": node_id,
        "output": output_key,
        "enabled": node._output_enabled[output_key],
    }


@router.post("/{node_id}/reload")
async def reload_node(node_id: str):
    """Hot-reload a dynamic node's code from disk."""
    from ...core.registry import _node_registry, remove_connection
    
    old_node = get_node(node_id)
    if not old_node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    # Check if node is dynamic
    if not getattr(old_node.__class__, "dynamic", False):
        raise HTTPException(status_code=400, detail="Node is not dynamic. Use @dynamic decorator on the class.")
    
    # Get the module containing this node class
    module_path = old_node.__class__.__module__
    module = sys.modules.get(module_path)
    
    if not module:
        raise HTTPException(status_code=500, detail=f"Module not found: {module_path}")
    
    try:
        # Reload the module to get updated class definition
        importlib.reload(module)
        new_class = getattr(module, old_node.__class__.__name__)
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Syntax error in module: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload module: {e}")
    
    # Verify the new class is still dynamic
    if not getattr(new_class, "dynamic", False):
        raise HTTPException(status_code=400, detail="Reloaded class is no longer dynamic")
    
    # Get new port names for connection validation
    new_input_ports = set(getattr(new_class, "_input_ports", {}).keys())
    new_output_ports = set(getattr(new_class, "_output_ports", {}).keys())
    
    # Remove invalid connections (ports that no longer exist)
    removed_connections = []
    for conn in get_connections_for_node(node_id):
        is_invalid = False
        # Check if connection involves a port that no longer exists
        if conn["from_node"] == node_id and conn["from_output"] not in new_output_ports:
            is_invalid = True
        if conn["to_node"] == node_id and conn["to_input"] not in new_input_ports:
            is_invalid = True
        
        if is_invalid:
            remove_connection(
                conn["from_node"], conn["from_output"],
                conn["to_node"], conn["to_input"]
            )
            removed_connections.append(conn)
    
    # Create new instance with same position
    new_node = new_class(
        x=old_node.position.get("x", 0),
        y=old_node.position.get("y", 0)
    )
    
    # Preserve node_id (keeps connections valid)
    new_node.node_id = node_id
    
    # Preserve name if it was customized
    new_node.name = old_node.name
    
    # Copy Store values (state preservation - these persist across reloads)
    for store_name in getattr(old_node.__class__, "_stores", {}).keys():
        if hasattr(old_node, store_name):
            try:
                setattr(new_node, store_name, getattr(old_node, store_name))
            except Exception as e:
                print(f"Warning: Could not copy store {store_name}: {e}")
    
    # Copy property values (user-adjusted settings)
    for prop_name in getattr(old_node.__class__, "_properties", {}).keys():
        if hasattr(old_node, prop_name):
            try:
                setattr(new_node, prop_name, getattr(old_node, prop_name))
            except Exception as e:
                print(f"Warning: Could not copy property {prop_name}: {e}")
    
    # Copy output enabled state
    if hasattr(old_node, "_output_enabled"):
        new_node._output_enabled = old_node._output_enabled.copy()
    
    # Replace in state.nodes list
    for i, n in enumerate(state.nodes):
        if n.node_id == node_id:
            state.nodes[i] = new_node
            break
    
    # Update registry
    _node_registry[node_id] = new_node
    
    # Update node_classes registry if needed
    if new_class.__name__ in state.node_classes:
        state.node_classes[new_class.__name__] = new_class
    
    # Initialize the new node (this is called fresh on reload)
    try:
        new_node.init()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize reloaded node: {e}")
    
    await broadcast_state()
    
    return {
        "status": "ok",
        "node": new_node.get_schema(),
        "message": f"Node {new_node.name} reloaded successfully",
        "removed_connections": removed_connections,
    }


@router.post("/rediscover")
async def rediscover_nodes():
    """
    Re-scan the nodes directory and register any new node classes.
    
    This allows adding new node types without restarting the server.
    """
    from ...core.descriptors.node import rediscover_nodes as do_rediscover
    
    try:
        # Re-discover all nodes
        new_palette = do_rediscover()
        
        # Update node_classes registry
        for category, classes in new_palette.items():
            for cls in classes:
                cls_name = cls.__name__
                if cls_name not in state.node_classes:
                    state.node_classes[cls_name] = cls
                    print(f"Registered new node class: {cls_name}")
        
        # Build the palette in the correct format for the UI
        panels = []
        for category, classes in new_palette.items():
            panel_nodes = []
            for cls in classes:
                panel_nodes.append({
                    "type": cls.__name__,
                    "name": cls.__name__,
                    "category": category,
                })
            panels.append({"name": category, "nodes": panel_nodes})
        
        # Update the node_palette in state
        state.node_palette = {"panels": panels}
        
        # Build response with discovered nodes
        discovered = {}
        for category, classes in new_palette.items():
            discovered[category] = [cls.__name__ for cls in classes]
        
        return {
            "status": "ok",
            "message": "Node discovery completed",
            "nodes": discovered,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rediscover nodes: {e}")
