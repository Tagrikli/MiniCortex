from typing import Any, Optional

from .base import BaseDescriptor


def _format_data_type(data_type: Any) -> str:
    if data_type is None:
        return "any"
    if isinstance(data_type, str):
        return data_type
    if isinstance(data_type, (list, tuple, set)):
        return " | ".join(_format_data_type(item) for item in data_type)
    if isinstance(data_type, type):
        if data_type.__module__ == "builtins":
            return data_type.__name__
        return f"{data_type.__module__}.{data_type.__name__}"
    return str(data_type)


class InputPort(BaseDescriptor):
    """
    Input port descriptor - represents a connection point for incoming data.
    
    Works like Property descriptors - values are stored on the instance and
    accessed directly via self.{attribute_name}.
    
    Example:
        class MyNode(Node):
            input_data = InputPort("Input", np.ndarray)
            
            def process(self):
                # Access directly via self.input_data
                if self.input_data is not None:
                    result = self.input_data * 2
    """
    
    def __init__(self, label: str, data_type: Any = "any", default: Any = None):
        super().__init__(label)
        self.data_type = data_type
        self.default = default
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)
    
    def __set__(self, obj, value):
        setattr(obj, f"_{self.name}", value)
    
    def to_spec(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "data_type": _format_data_type(self.data_type),
        }


class OutputPort(BaseDescriptor):
    """
    Output port descriptor - represents a connection point for outgoing data.
    
    Works like Property descriptors - values are stored on the instance and
    accessed directly via self.{attribute_name}.
    
    Example:
        class MyNode(Node):
            output_data = OutputPort("Output", np.ndarray)
            
            def process(self):
                # Set directly via self.output_data
                self.output_data = self.input_data * 2
            
            def output(self):
                # Return dict with port values (can be auto-generated)
                return {"output_data": self.output_data}
    """
    
    def __init__(self, label: str, data_type: Any = "any", default: Any = None):
        super().__init__(label)
        self.data_type = data_type
        self.default = default
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)
    
    def __set__(self, obj, value):
        setattr(obj, f"_{self.name}", value)
    
    def to_spec(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "data_type": _format_data_type(self.data_type),
        }
