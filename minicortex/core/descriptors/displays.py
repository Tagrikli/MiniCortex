from typing import Any
from .base import Display

class Numeric(Display):
    """Scalar output display descriptor."""
    def __init__(self, label: str, format: str = ".4f"):
        super().__init__(label, 0.0)
        self.format = format

    def to_spec(self, value: Any = None) -> dict:
        formatted = f"{value:{self.format}}" if isinstance(value, (int, float)) else str(value)
        return {
            "type": "numeric",
            "label": self.label,
            "format": self.format,
            "value": value,
            "formatted": formatted,
        }


class Vector1D(Display):
    """1D array visualization descriptor."""
    def __init__(self, label: str, color_mode: str = "grayscale"):
        super().__init__(label)
        self.color_mode = color_mode

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "vector1d",
            "label": self.label,
            "color_mode": self.color_mode,
            "shape": list(value.shape) if value is not None and hasattr(value, "shape") else None,
        }


class Vector2D(Display):
    """2D array visualization descriptor."""
    def __init__(self, label: str, color_mode: str = "grayscale"):
        super().__init__(label)
        self.color_mode = color_mode

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "vector2d",
            "label": self.label,
            "color_mode": self.color_mode,
            "shape": list(value.shape) if value is not None and hasattr(value, "shape") else None,
        }


class Text(Display):
    """Text output display descriptor."""
    def __init__(self, label: str, default: str = ""):
        super().__init__(label, default)

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "text",
            "label": self.label,
            "default": self.default,
            "value": value or self.default,
        }
