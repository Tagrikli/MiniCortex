"""
MiniCortex server routes.
"""

from .nodes import router as nodes_router
from .connections import router as connections_router
from .network import router as network_router
from .workspaces import router as workspaces_router

__all__ = ["nodes_router", "connections_router", "network_router", "workspaces_router"]
