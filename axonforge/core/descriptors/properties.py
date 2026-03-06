from typing import Any, Callable, List, Optional, Union
from .base import Property


class Range(Property[float]):
    """Numeric range property descriptor with min/max bounds."""

    def __init__(
        self,
        label: str,
        default: float,
        min_val: float,
        max_val: float,
        step: float = 0.01,
        scale: str = "linear",
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, default, on_change)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.scale = scale

    def validate(self, value: float) -> None:
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(
                f"{self.name} must be in [{self.min_val}, {self.max_val}], got {value}"
            )

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "range",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
            "min": self.min_val,
            "max": self.max_val,
            "step": self.step,
            "scale": self.scale,
        }


class Integer(Property[int]):
    """Integer property descriptor."""

    def __init__(
        self,
        label: str,
        default: int = 0,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, int(default), on_change)

    def validate(self, value: int) -> None:
        try:
            int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name} must be an integer, got {value}")

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "integer",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
        }


class Float(Property[float]):
    """Float property descriptor."""

    def __init__(
        self,
        label: str,
        default: float = 0.0,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, float(default), on_change)

    def validate(self, value: float) -> None:
        try:
            float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name} must be a float, got {value}")

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "float",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
        }


class Bool(Property[bool]):
    """Boolean property descriptor."""

    def __init__(
        self,
        label: str,
        default: bool = False,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, default, on_change)

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "bool",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
        }


class Enum(Property[str]):
    """Enum selection property descriptor."""

    def __init__(
        self,
        label: str,
        options: List[str],
        default: str,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, default, on_change)
        self.options = options

    def validate(self, value: str) -> None:
        if value not in self.options:
            raise ValueError(f"{self.name} must be one of {self.options}, got {value}")

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "enum",
            "label": self.label,
            "options": self.options,
            "default": self.default,
            "value": value if value is not None else self.default,
        }
