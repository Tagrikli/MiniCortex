"""
Node category decorators for auto-registration in the node palette.

Usage:
    from minicortex.core.decorators import node
    
    @node.input
    class MyInputNode(Node):
        ...
    
    @node.utility
    class MyUtilityNode(Node):
        ...
    
    @node.processing
    class MyProcessingNode(Node):
        ...
"""

from typing import Dict, List, Type, Optional
import importlib
import sys
from pathlib import Path


# Category registry: {"Input": [Class1, Class2], "Utilities": [...], ...}
_node_categories: Dict[str, List[Type]] = {}

# Track discovered modules to avoid re-registering
_discovered_modules: set = set()


def get_node_categories() -> Dict[str, List[Type]]:
    """Get all registered node categories and their classes."""
    return _node_categories.copy()


def get_all_node_classes() -> List[Type]:
    """Get all registered node classes across all categories."""
    all_classes = []
    for classes in _node_categories.values():
        all_classes.extend(classes)
    return all_classes


def build_node_palette() -> Dict[str, List[Type]]:
    """Build the node_palette dict from registered categories."""
    return _node_categories.copy()


def clear_node_registry():
    """Clear all registered nodes (for re-discovery)."""
    global _node_categories, _discovered_modules
    _node_categories = {}
    _discovered_modules = set()


def discover_nodes(nodes_dir = None, package: str = "minicortex.nodes") -> Dict[str, List[Type]]:
    """
    Discover all node classes in the nodes directory.
    
    Scans Python files in the nodes directory, imports them, and registers
    any classes decorated with @node.input, @node.utility, etc.
    
    Args:
        nodes_dir: Directory to scan (defaults to minicortex/nodes)
        package: Package name for imports
    
    Returns:
        Dictionary of discovered node categories
    """
    if nodes_dir is None:
        # Find the nodes directory relative to this file
        nodes_path: Path = Path(__file__).parent.parent.parent / "nodes"
    else:
        nodes_path = Path(nodes_dir) if isinstance(nodes_dir, str) else nodes_dir
    
    if not nodes_path.exists():
        print(f"Warning: Nodes directory not found: {nodes_path}")
        return _node_categories.copy()
    
    # Get all Python files in the nodes directory
    py_files = list(nodes_path.glob("*.py"))
    
    for py_file in py_files:
        # Skip __init__.py and files starting with _
        if py_file.name.startswith("_"):
            continue
        
        module_name = py_file.stem
        full_module_name = f"{package}.{module_name}"
        
        # Skip already discovered modules
        if full_module_name in _discovered_modules:
            continue
        
        try:
            # Import the module to trigger decorator registration
            if full_module_name in sys.modules:
                # Reload if already imported
                importlib.reload(sys.modules[full_module_name])
            else:
                importlib.import_module(full_module_name)
            
            _discovered_modules.add(full_module_name)
            print(f"Discovered node module: {full_module_name}")
        except Exception as e:
            print(f"Warning: Failed to import {full_module_name}: {e}")
    
    return _node_categories.copy()


def rediscover_nodes(nodes_dir: Optional[str] = None, package: str = "minicortex.nodes") -> Dict[str, List[Type]]:
    """
    Re-discover all nodes by clearing registry and re-scanning.
    
    This is useful for hot-reloading new node definitions without
    restarting the server.
    
    Args:
        nodes_dir: Directory to scan (defaults to minicortex/nodes)
        package: Package name for imports
    
    Returns:
        Dictionary of discovered node categories
    """
    clear_node_registry()
    return discover_nodes(nodes_dir, package)


def _create_category_decorator(category: str, label: Optional[str] = None):
    """
    Factory that creates a decorator for a specific node category.
    
    Args:
        category: The category name (e.g., "Input", "Utilities", "Processing")
        label: Optional custom label for the node in the palette
    
    Returns:
        A decorator function that registers the class in the category
    """
    def decorator(cls: Type) -> Type:
        # Register in category
        if category not in _node_categories:
            _node_categories[category] = []
        
        # Check if already registered (avoid duplicates on reload)
        if cls not in _node_categories[category]:
            _node_categories[category].append(cls)
        
        # Mark with category metadata
        cls._node_category = category
        
        # Store optional custom label
        if label:
            cls._node_label = label
        
        return cls
    
    return decorator


class NodeDecorators:
    """
    Namespace for node category decorators.
    
    Usage:
        @node.input
        class MyInput(Node): ...
        
        @node.utility
        class MyUtility(Node): ...
        
        @node.processing
        class MyProcessing(Node): ...
    """
    
    @staticmethod
    def input(cls: Type) -> Type:
        """Decorator for input nodes (data sources, generators)."""
        return _create_category_decorator("Input")(cls)
    
    @staticmethod
    def utility(cls: Type) -> Type:
        """Decorator for utility nodes (data transformation, display)."""
        return _create_category_decorator("Utilities")(cls)
    
    @staticmethod
    def processing(cls: Type) -> Type:
        """Decorator for processing nodes (algorithms, computation)."""
        return _create_category_decorator("Processing")(cls)
    
    @staticmethod
    def output(cls: Type) -> Type:
        """Decorator for output nodes (data sinks, exporters)."""
        return _create_category_decorator("Output")(cls)
    
    @staticmethod
    def custom(category: str, label: Optional[str] = None):
        """
        Decorator for custom categories.
        
        Usage:
            @node.custom("Experimental")
            class MyExperimentalNode(Node): ...
        """
        return _create_category_decorator(category, label)


# Create the singleton namespace object
node = NodeDecorators()
