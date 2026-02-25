"""
Connection API routes.
"""

from fastapi import APIRouter, HTTPException

from ..state import state
from ..models import ConnectionCreate
from ..websocket import broadcast_state
from ...core.registry import (
    get_node, get_connections, add_connection, remove_connection
)
from ...core.descriptors.ports import _format_data_type
from .nodes import update_editor_viewport, build_topology_snapshot


router = APIRouter(prefix="/api/connections", tags=["connections"])


def _is_type_compatible(output_type, input_type) -> bool:
    """Check if two port types are compatible."""
    def is_any(t):
        return t is None or (isinstance(t, str) and t.lower() == "any")

    if is_any(input_type) or is_any(output_type):
        return True
    if isinstance(input_type, (list, tuple, set)):
        return any(_is_type_compatible(output_type, opt) for opt in input_type)
    if isinstance(output_type, (list, tuple, set)):
        return any(_is_type_compatible(opt, input_type) for opt in output_type)
    if isinstance(output_type, type) and isinstance(input_type, type):
        return issubclass(output_type, input_type)
    return _format_data_type(output_type) == _format_data_type(input_type)


@router.post("")
async def create_connection(body: ConnectionCreate):
    """Create a connection between nodes."""
    from_node, to_node = get_node(body.from_node), get_node(body.to_node)
    if not from_node or not to_node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Find ports
    from_port = next(
        (p for p in getattr(from_node.__class__, "_output_ports", {}).values()
         if p.name == body.from_output),
        None,
    )
    to_port = next(
        (p for p in getattr(to_node.__class__, "_input_ports", {}).values()
         if p.name == body.to_input),
        None,
    )
    
    if not from_port or not to_port:
        raise HTTPException(status_code=404, detail="Port not found")
    
    # Check type compatibility
    if not _is_type_compatible(from_port.data_type, to_port.data_type):
        raise HTTPException(status_code=400, detail="Incompatible types")
    
    # Remove existing connection to the same input and clear stale value
    for conn in get_connections():
        if conn["to_node"] == body.to_node and conn["to_input"] == body.to_input:
            remove_connection(
                conn["from_node"], conn["from_output"],
                conn["to_node"], conn["to_input"],
            )
            # Clear the stale value from the old connection
            old_signal_key = (conn["from_node"], conn["from_output"])
            if state.network and hasattr(state.network, "_signals_now"):
                if old_signal_key in state.network._signals_now:
                    del state.network._signals_now[old_signal_key]
    
    # Add new connection
    if not add_connection(body.from_node, body.from_output, body.to_node, body.to_input):
        raise HTTPException(status_code=400, detail="Connection exists")
    
    # Notify target node if it has on_connect handler
    if hasattr(to_node, "on_connect"):
        try:
            # Get the output value from the OutputPort descriptor
            output_value = getattr(from_node, body.from_output, None)
            to_node.on_connect(
                from_node=from_node,
                from_output=body.from_output,
                to_input=body.to_input,
                value=output_value,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Propagate data through the new connection
    # This processes the target node and all downstream nodes
    if state.network:
        state.network.propagate_from_node(body.to_node, recompute_start=True)
    
    update_editor_viewport(body.viewport)
    await broadcast_state()
    
    return {"status": "ok", "snapshot": build_topology_snapshot()}


@router.delete("")
async def delete_connection(body: ConnectionCreate):
    """Delete a connection between nodes."""
    if not remove_connection(body.from_node, body.from_output, body.to_node, body.to_input):
        raise HTTPException(status_code=404, detail="Connection not found")
    
    # Clear the stale value from the target node's input port
    target_node = get_node(body.to_node)
    if target_node and hasattr(target_node, body.to_input):
        setattr(target_node, body.to_input, None)
    
    # Also clear from network's signal cache if present
    if state.network:
        signal_key = (body.from_node, body.from_output)
        if hasattr(state.network, "_signals_now") and signal_key in state.network._signals_now:
            del state.network._signals_now[signal_key]
    
    update_editor_viewport(body.viewport)
    await broadcast_state()
    
    return {"status": "ok", "snapshot": build_topology_snapshot()}
