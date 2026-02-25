from typing import Any, Callable, List, Optional, Union
from .base import Property

class Slider(Property):
    """Numeric property descriptor with min/max bounds."""
    def __init__(
        self,
        label: str,
        default: float,
        min_val: float,
        max_val: float,
        scale: str = "linear",
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, default, on_change)
        self.min_val = min_val
        self.max_val = max_val
        self.scale = scale

    def validate(self, value):
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(
                f"{self.name} must be in [{self.min_val}, {self.max_val}], got {value}"
            )

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "slider",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
            "min": self.min_val,
            "max": self.max_val,
            "scale": self.scale,
        }


class Integer(Property):
    """Integer property descriptor."""
    def __init__(
        self,
        label: str,
        default: int = 0,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, int(default), on_change)
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, value):
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name} must be an integer, got {value}")
        
        if self.min_val is not None and int_value < self.min_val:
            raise ValueError(f"{self.name} must be >= {self.min_val}, got {int_value}")
            
        if self.max_val is not None and int_value > self.max_val:
            raise ValueError(f"{self.name} must be <= {self.max_val}, got {int_value}")

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "integer",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
            "min": self.min_val,
            "max": self.max_val,
        }


class CheckBox(Property):
    """Boolean property descriptor."""
    def __init__(self, label: str, default: bool = False, on_change: Union[str, Callable, None] = None):
        super().__init__(label, default, on_change)

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "checkbox",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
        }


class RadioButtons(Property):
    """Enum selection property descriptor."""
    def __init__(self, label: str, options: List[str], default: str, on_change: Union[str, Callable, None] = None):
        super().__init__(label, default, on_change)
        self.options = options

    def validate(self, value):
        if value not in self.options:
            raise ValueError(f"{self.name} must be one of {self.options}, got {value}")

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "radio",
            "label": self.label,
            "options": self.options,
            "default": self.default,
            "value": value if value is not None else self.default,
        }
