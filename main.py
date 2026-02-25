"""
main.py - Entry point for MiniCortex Node Editor.
"""

from minicortex.network.network import Network
from minicortex.server.server import init_server, run_server
from minicortex.core.descriptors.node import discover_nodes, build_node_palette


def create_network() -> Network:
    """Create the network manager."""
    return Network()


def main():
    """Main entry point."""
    print("MiniCortex - Node Editor")
    print("=" * 40)

    # Create network manager
    network = create_network()

    # Discover all node classes automatically
    discover_nodes()
    
    # Build node palette from discovered nodes
    node_palette = build_node_palette()

    print(f"\nNode classes registered:")
    for category, classes in node_palette.items():
        for cls in classes:
            node_type = cls.__name__
            print(f"  - {node_type} ({category})")

    # Initialize server with nodes, network, and auto-built node palette
    init_server(nodes=[], network=network, ws_fps=40, node_palette=node_palette)

    print(f"\nServer starting at http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    # Run server
    run_server()


if __name__ == "__main__":
    main()
