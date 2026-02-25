"""
WebSocket handling for MiniCortex server.
"""

import json
from typing import Dict, Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .state import state


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_display_outputs(node) -> Dict[str, Any]:
    """Build outputs dictionary from Display descriptors only."""
    outputs = {}
    for name in getattr(node.__class__, "_outputs", {}).keys():
        try:
            val = getattr(node, name)
        except Exception:
            val = None
        outputs[name] = val
    return outputs


def build_network_state() -> Dict[str, Any]:
    """Get current network state for WebSocket messages."""
    if state.network is None:
        return {"running": False, "speed": 10, "step": 0, "actual_hz": 0.0}
    return {
        "running": bool(getattr(state.network, "running", False)),
        "speed": float(getattr(state.network, "speed", 10)),
        "step": (
            int(state.network.get_step_count()) 
            if hasattr(state.network, "get_step_count") else 0
        ),
        "actual_hz": float(getattr(state.network, "actual_hz", 0.0)),
    }


async def send_state_to_client(websocket: WebSocket):
    """Send current state to a single WebSocket client."""
    nodes_payload = {}
    for node in state.nodes:
        nodes_payload[node.node_id] = {"outputs": build_display_outputs(node)}
    data = {"type": "state", "nodes": nodes_payload}
    await websocket.send_text(json.dumps(data, cls=NumpyEncoder))


async def broadcast_state():
    """Broadcast current state to all connected WebSocket clients."""
    if not state.websocket_clients:
        return
    try:
        nodes_payload = {}
        for node in state.nodes:
            nodes_payload[node.node_id] = {"outputs": build_display_outputs(node)}
        data = {
            "type": "state", 
            "nodes": nodes_payload, 
            "network": build_network_state()
        }
        msg = json.dumps(data, cls=NumpyEncoder)
        for client in list(state.websocket_clients):
            try:
                await client.send_text(msg)
            except:
                state.websocket_clients.discard(client)
    except Exception as e:
        print(f"Broadcast error: {e}")


async def broadcast_error(node_id: str, node_name: str, error: str, traceback: str):
    """Broadcast an error to all connected WebSocket clients."""
    if not state.websocket_clients:
        return
    try:
        data = {
            "type": "error",
            "node_id": node_id,
            "node_name": node_name,
            "error": error,
            "traceback": traceback,
            "network": build_network_state(),  # Include network state (will show stopped)
        }
        msg = json.dumps(data, cls=NumpyEncoder)
        for client in list(state.websocket_clients):
            try:
                await client.send_text(msg)
            except:
                state.websocket_clients.discard(client)
    except Exception as e:
        print(f"Broadcast error: {e}")


async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection lifecycle."""
    await websocket.accept()
    state.websocket_clients.add(websocket)
    try:
        await send_state_to_client(websocket)
        while True:
            data = await websocket.receive_text()
            try:
                if json.loads(data).get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except:
                pass
    except WebSocketDisconnect:
        state.websocket_clients.discard(websocket)
