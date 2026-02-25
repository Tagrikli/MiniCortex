from typing import Any, Dict, Optional
import json
import numpy as np

from .descriptors.base import Property, Display, Action
from .descriptors.ports import InputPort, OutputPort
from .descriptors.store import Store
from .registry import get_connections_for_node


class NodeMeta(type):
    """Metaclass that collects inputs/outputs/properties/actions from class attributes."""
    
    def __new__(mcs, name, bases, namespace):
        # Start with inherited registries so subclasses can extend/override.
        _input_ports = {}
        _output_ports = {}
        _properties = {}
        _actions = {}
        _outputs = {}
        _stores = {}
        for base in reversed(bases):
            _input_ports.update(getattr(base, "_input_ports", {}))
            _output_ports.update(getattr(base, "_output_ports", {}))
            _properties.update(getattr(base, "_properties", {}))
            _actions.update(getattr(base, "_actions", {}))
            _outputs.update(getattr(base, "_outputs", {}))
            _stores.update(getattr(base, "_stores", {}))
        
        # Scan namespace for descriptors
        for attr_name, attr_value in namespace.items():
            if attr_name.startswith('_'):
                continue
            
            # Categories based on base classes
            if isinstance(attr_value, InputPort):
                _input_ports[attr_name] = attr_value
            elif isinstance(attr_value, OutputPort):
                _output_ports[attr_name] = attr_value
            elif isinstance(attr_value, Property):
                _properties[attr_name] = attr_value
            elif isinstance(attr_value, Action):
                _actions[attr_name] = attr_value
            elif isinstance(attr_value, Display):
                _outputs[attr_name] = attr_value
            elif isinstance(attr_value, Store):
                _stores[attr_name] = attr_value
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Attach registries to the class
        cls._input_ports = _input_ports
        cls._output_ports = _output_ports
        cls._properties = _properties
        cls._actions = _actions
        cls._outputs = _outputs
        cls._stores = _stores
        
        return cls


class Node(metaclass=NodeMeta):
    """
    Base class for all node types.
    
    InputPort and OutputPort values are accessed directly via self.{port_name}.
    Property values are accessed directly via self.{property_name}.
    Display values are accessed directly via self.{display_name}.
    
    Example:
        class MyNode(Node):
            input_data = InputPort("Input", np.ndarray)
            output_data = OutputPort("Output", np.ndarray)
            threshold = Slider("Threshold", 0.5, 0.0, 1.0)
            result = Vector2D("Result")
            
            def process(self):
                # Access inputs directly
                if self.input_data is not None:
                    # Set outputs directly
                    self.output_data = self.input_data * self.threshold
                    self.result = self.output_data
    """
    
    # Set to True in subclasses to enable hot-reload functionality
    dynamic: bool = False
    
    def __init__(self, x: float = 0, y: float = 0):
        self.node_id: Optional[str] = None
        # Use class name as both stable type identifier and default display name.
        self.node_type: str = self.__class__.__name__
        self.position: Dict[str, float] = {"x": x, "y": y}
        self.name: str = self.__class__.__name__
        
        # State
        self._output_enabled: Dict[str, bool] = {
            name: True for name in getattr(self.__class__, "_outputs", {}).keys()
        }

    def init(self):
        """Called once after initialization and registration."""
        pass

    def process(self):
        """
        The main compute method for the node.
        
        Access inputs via self.{input_port_name}.
        Set outputs via self.{output_port_name}.
        """
        raise NotImplementedError("Each node must implement process")

    def validate_required_methods(self):
        """Validate that the node has all required methods."""
        required = ["process"]
        for method_name in required:
            if not hasattr(self, method_name):
                raise AttributeError(f"Node {self.__class__.__name__} missing required method: {method_name}")

    def get_schema(self) -> dict:
        """
        Return the node's schema for the UI.
        """
        # Input/output ports (for connections UI)
        input_ports = [p.to_spec() for p in getattr(self.__class__, "_input_ports", {}).values()]
        output_ports = [p.to_spec() for p in getattr(self.__class__, "_output_ports", {}).values()]

        # Properties (editable controls in the node body)
        properties = []
        for name, prop in getattr(self.__class__, "_properties", {}).items():
            val = getattr(self, name)
            spec = prop.to_spec(val)
            spec["key"] = name
            properties.append(spec)

        # Actions (buttons)
        actions = []
        for name, action in getattr(self.__class__, "_actions", {}).items():
            spec = action.to_spec()
            spec["key"] = name
            actions.append(spec)

        # Output displays (visual/text outputs shown in the node body)
        outputs = []
        for name, display in getattr(self.__class__, "_outputs", {}).items():
            val = getattr(self, name)
            spec = display.to_spec(val)
            spec["key"] = name
            spec["enabled"] = self._output_enabled.get(name, True)
            outputs.append(spec)

        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "name": self.name,
            "position": self.position,
            "dynamic": getattr(self.__class__, "dynamic", False),
            "input_ports": input_ports,
            "output_ports": output_ports,
            "properties": properties,
            "actions": actions,
            "outputs": outputs,
        }

    def to_dict(self) -> dict:
        """
        Return a serializable representation of the node for persistence.
        """
        data = {
            "id": self.node_id,
            "type": self.node_type,
            "name": self.name,
            "position": self.position,
            "properties": {},
            "stores": {},
        }
        
        # Save all property values
        for prop_name in getattr(self.__class__, "_properties", {}).keys():
            val = getattr(self, prop_name)
            # Handle non-serializable values (basic approach)
            if isinstance(val, (int, float, str, bool, list, dict)) or val is None:
                data["properties"][prop_name] = val
            else:
                data["properties"][prop_name] = str(val)
        
        # Save all store values
        for store_name in getattr(self.__class__, "_stores", {}).keys():
            val = getattr(self, store_name)
            # Handle numpy arrays by converting to list
            if isinstance(val, np.ndarray):
                data["stores"][store_name] = {
                    "_type": "ndarray",
                    "data": val.tolist(),
                }
            elif isinstance(val, (int, float, str, bool, list, dict)) or val is None:
                data["stores"][store_name] = val
            else:
                # Try to serialize as string for other types
                data["stores"][store_name] = str(val)
                
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """
        Create a node instance from a dictionary representation.
        """
        node = cls(
            x=data.get("position", {}).get("x", 0),
            y=data.get("position", {}).get("y", 0)
        )
        node.node_id = data.get("id")
        node.name = data.get("name", cls.__name__)
        
        # Restore property values
        for prop_name, prop_val in data.get("properties", {}).items():
            if hasattr(node, prop_name):
                try:
                    setattr(node, prop_name, prop_val)
                except Exception as e:
                    print(f"Error restoring property {prop_name}: {e}")
        
        # Restore store values
        for store_name, store_val in data.get("stores", {}).items():
            if hasattr(node, store_name):
                try:
                    # Handle numpy arrays
                    if isinstance(store_val, dict) and store_val.get("_type") == "ndarray":
                        store_val = np.array(store_val["data"])
                    setattr(node, store_name, store_val)
                except Exception as e:
                    print(f"Error restoring store {store_name}: {e}")
                    
        return node
