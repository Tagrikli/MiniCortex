"""
Pydantic request/response models for MiniCortex API.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class PropertyUpdate(BaseModel):
    """Request body for updating a node property."""
    value: Any


class ActionRequest(BaseModel):
    """Request body for executing a node action."""
    params: Optional[dict] = None


class PositionUpdate(BaseModel):
    """Request body for updating node position."""
    x: float
    y: float


class PanState(BaseModel):
    """Editor pan state."""
    x: float
    y: float


class ViewportState(BaseModel):
    """Editor viewport state."""
    pan: PanState
    zoom: float


class TopologyContext(BaseModel):
    """Optional topology context for requests."""
    viewport: Optional[ViewportState] = None


class NetworkSpeedUpdate(BaseModel):
    """Request body for updating network speed."""
    speed: float


class ConnectionCreate(BaseModel):
    """Request body for creating a connection."""
    from_node: str
    from_output: str
    to_node: str
    to_input: str
    viewport: Optional[ViewportState] = None


class NodeCreate(BaseModel):
    """Request body for creating a node."""
    type: str
    position: Dict[str, float]
    viewport: Optional[ViewportState] = None


class OutputEnableUpdate(BaseModel):
    """Request body for enabling/disabling an output."""
    enabled: bool
