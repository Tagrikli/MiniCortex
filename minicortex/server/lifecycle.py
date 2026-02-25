"""
Background loops and application lifecycle for MiniCortex server.
"""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .state import state
from .websocket import broadcast_state, broadcast_error


async def computation_loop():
    """Background loop for network computation."""
    while True:
        if state.network and getattr(state.network, "running", False):
            try:
                if hasattr(state.network, "execute_step"):
                    state.network.execute_step()
                
                now = time.monotonic()
                if state.network_last_step_time:
                    dt = now - state.network_last_step_time
                    if dt > 0:
                        state.network_actual_hz = 1.0 / dt
                        setattr(state.network, "actual_hz", state.network_actual_hz)
                state.network_last_step_time = now
                
                speed = float(getattr(state.network, "speed", 10))
                await asyncio.sleep(
                    0 if speed >= state.network_max_hz else 1.0 / max(speed, 1.0)
                )
            except Exception as e:
                # Import NetworkError to check if it's our error type
                from ..network.network import NetworkError
                if isinstance(e, NetworkError):
                    print(f"Network error in node '{e.node_name}': {e.error}")
                    print(f"Traceback:\n{e.traceback}")
                    # Broadcast error to all connected clients
                    await broadcast_error(
                        node_id=e.node_id,
                        node_name=e.node_name,
                        error=str(e.error),
                        traceback=e.traceback
                    )
                else:
                    print(f"Computation error: {e}")
                    # Stop network on any error
                    if state.network:
                        state.network.running = False
                await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.05)


async def broadcast_loop():
    """Background loop for WebSocket broadcasts."""
    while True:
        try:
            await broadcast_state()
        except Exception as e:
            print(f"Broadcast error: {e}")
        await asyncio.sleep(1.0 / state.ws_fps if state.ws_fps > 0 else 0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    if state.computation_task is None:
        state.computation_task = asyncio.create_task(computation_loop())
    if state.broadcast_task is None:
        state.broadcast_task = asyncio.create_task(broadcast_loop())
    
    yield
    
    # Shutdown
    if state.computation_task:
        state.computation_task.cancel()
        try:
            await state.computation_task
        except asyncio.CancelledError:
            pass
    if state.broadcast_task:
        state.broadcast_task.cancel()
        try:
            await state.broadcast_task
        except asyncio.CancelledError:
            pass
