"""
Network control API routes.
"""

from fastapi import APIRouter

from ..state import state, get_network_max_hz
from ..models import NetworkSpeedUpdate
from ..websocket import broadcast_state


router = APIRouter(prefix="/api/network", tags=["network"])


def get_network_state() -> dict:
    """Get current network state."""
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


@router.post("/start")
async def start_network():
    """Start the network execution."""
    if state.network and hasattr(state.network, "start"):
        state.network.start()
    return {"status": "ok", "network": get_network_state()}


@router.post("/stop")
async def stop_network():
    """Stop the network execution."""
    if state.network and hasattr(state.network, "stop"):
        state.network.stop()
    return {"status": "ok", "network": get_network_state()}


@router.post("/step")
async def step_network():
    """Execute a single network step."""
    if state.network:
        if getattr(state.network, "running", False) and hasattr(state.network, "stop"):
            state.network.stop()
        if hasattr(state.network, "execute_step"):
            state.network.execute_step()
    await broadcast_state()
    return {"status": "ok", "network": get_network_state()}


@router.put("/speed")
async def set_network_speed(body: NetworkSpeedUpdate):
    """Set network execution speed."""
    if state.network:
        setattr(
            state.network, "speed", 
            max(1.0, min(get_network_max_hz(), float(body.speed)))
        )
    return {"status": "ok", "network": get_network_state()}
