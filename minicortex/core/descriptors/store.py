from typing import Any, Optional
from .base import BaseDescriptor


class Store(BaseDescriptor):
    """
    Descriptor for marking instance variables that should be persisted.
    
    Works like other descriptors - values are stored on the instance and
    accessed directly via self.{attribute_name}.
    
    Example:
        class MyNode(Node):
            angle = Store(default=0.0)
            matrix = Store(default=None)
            
            def process(self):
                # Access directly
                self.angle += 0.1
    """
    
    def __init__(self, label: Optional[str] = None, default: Any = None):
        super().__init__(label or "")
        self.default = default
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)
    
    def __set__(self, obj, value):
        setattr(obj, f"_{self.name}", value)
    
    def to_spec(self, value: Any = None) -> dict:
        """Return a specification dict for this store."""
        return {
            "name": self.name,
            "label": self.label or self.name,
            "default": self.default,
        }
