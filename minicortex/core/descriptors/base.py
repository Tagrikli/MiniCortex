from typing import Any, Callable, Optional, Union

class BaseDescriptor:
    """Base class for all node metadata descriptors."""
    def __init__(self, label: str):
        self.label = label
        self.name: Optional[str] = None

    def __set_name__(self, owner, name: str):
        self.name = name

    def to_spec(self, value: Any = None) -> dict:
        raise NotImplementedError


class Property(BaseDescriptor):
    """Base for interactive property descriptors bound to instance variables."""
    def __init__(self, label: str, default: Any, on_change: Union[str, Callable, None] = None):
        super().__init__(label)
        self.default = default
        self.on_change = on_change

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)

    def __set__(self, obj, value):
        old = getattr(obj, f"_{self.name}", self.default)
        self.validate(value)
        setattr(obj, f"_{self.name}", value)
        if value != old and self.on_change:
            self._trigger_change(obj, value, old)

    def validate(self, value):
        """Override in subclasses to add validation."""
        pass

    def _trigger_change(self, obj, new_value, old_value):
        on_change = self.on_change
        if isinstance(on_change, str):
            callback = getattr(obj, on_change, None)
            if callable(callback):
                callback(new_value, old_value)
        elif callable(on_change):
            on_change(obj, new_value, old_value)


class Display(BaseDescriptor):
    """Base for display-only output descriptors bound to instance variables."""
    def __init__(self, label: str, default: Any = None):
        super().__init__(label)
        self.default = default

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)

    def __set__(self, obj, value):
        setattr(obj, f"_{self.name}", value)



