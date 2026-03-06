from typing import Any, Callable, Optional, Union, TypeVar, Generic, overload

T = TypeVar("T")


class BaseDescriptor:
    """Base class for all node metadata descriptors."""

    def __init__(self, label: str):
        self.label = label
        self.name: Optional[str] = None

    def __set_name__(self, owner, name: str):
        self.name = name

    def to_spec(self, value: Any = None) -> dict:
        raise NotImplementedError


class Property(BaseDescriptor, Generic[T]):
    """Base for interactive property descriptors bound to instance variables."""

    def __init__(
        self,
        label: str,
        default: T,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label)
        self.default = default
        self.on_change = on_change

    @overload
    def __get__(self, obj: None, objtype: type) -> "Property[T]": ...

    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)

    def __set__(self, obj, value: T) -> None:
        old = getattr(obj, f"_{self.name}", self.default)
        self.validate(value)
        setattr(obj, f"_{self.name}", value)
        if value != old and self.on_change:
            self._trigger_change(obj, value, old)

    def validate(self, value: T) -> None:
        """Override in subclasses to add validation."""
        pass

    def _trigger_change(self, obj, new_value: T, old_value: T) -> None:
        on_change = self.on_change
        if isinstance(on_change, str):
            callback = getattr(obj, on_change, None)
            if callable(callback):
                callback(new_value, old_value)
        elif callable(on_change):
            on_change(obj, new_value, old_value)


class Display(BaseDescriptor, Generic[T]):
    """Base for display-only output descriptors bound to instance variables."""

    def __init__(
        self,
        label: str,
        default: Optional[T] = None,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label)
        self.default = default
        self.on_change = on_change

    @overload
    def __get__(self, obj: None, objtype: type) -> "Display[T]": ...

    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)

    def __set__(self, obj, value: T) -> None:
        setattr(obj, f"_{self.name}", value)

    def _trigger_change(self, obj, new_config: dict, old_config: dict):
        """Trigger on_change callback when display config changes."""
        on_change = self.on_change
        if isinstance(on_change, str):
            callback = getattr(obj, on_change, None)
            if callable(callback):
                callback(new_config, old_config)
        elif callable(on_change):
            on_change(obj, new_config, old_config)
