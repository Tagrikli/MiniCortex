from typing import Any, Callable, Optional, Union
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


class BarChart(Display):
    """1D array bar chart visualization descriptor."""
    def __init__(
        self,
        label: str,
        color: str = "#e94560",
        show_negative: bool = True,
        scale_mode: str = "auto",
        scale_min: float = 0.0,
        scale_max: float = 1.0,
    ):
        super().__init__(label)
        self.color = color
        self.show_negative = show_negative
        self.scale_mode = scale_mode
        self.scale_min = scale_min
        self.scale_max = scale_max

    def change_scale(self, scale_mode: str, scale_min: Optional[float] = None, scale_max: Optional[float] = None, obj: Any = None) -> None:
        """Change scale settings at runtime and notify the UI.
        
        Args:
            scale_mode: The new scale mode ("auto" or "manual")
            scale_min: The minimum value for manual scaling (optional)
            scale_max: The maximum value for manual scaling (optional)
            obj: The node instance to pass to the on_change callback (optional)
        """
        old_config = {
            "scale_mode": self.scale_mode,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
        }
        self.scale_mode = scale_mode
        if scale_min is not None:
            self.scale_min = scale_min
        if scale_max is not None:
            self.scale_max = scale_max
        # Trigger on_change callback if configured
        if self.on_change and obj is not None:
            self._trigger_change(obj, {
                "scale_mode": self.scale_mode,
                "scale_min": self.scale_min,
                "scale_max": self.scale_max,
            }, old_config)

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "barchart",
            "label": self.label,
            "color": self.color,
            "show_negative": self.show_negative,
            "scale_mode": self.scale_mode,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "shape": list(value.shape) if value is not None and hasattr(value, "shape") else None,
        }


class LineChart(Display):
    """1D array line chart visualization descriptor."""
    def __init__(
        self,
        label: str,
        color: str = "#00f5ff",
        line_width: float = 1.5,
        scale_mode: str = "auto",
        scale_min: float = 0.0,
        scale_max: float = 1.0,
    ):
        super().__init__(label)
        self.color = color
        self.line_width = line_width
        self.scale_mode = scale_mode
        self.scale_min = scale_min
        self.scale_max = scale_max

    def change_scale(self, scale_mode: str, scale_min: Optional[float] = None, scale_max: Optional[float] = None, obj: Any = None) -> None:
        """Change scale settings at runtime and notify the UI.
        
        Args:
            scale_mode: The new scale mode ("auto" or "manual")
            scale_min: The minimum value for manual scaling (optional)
            scale_max: The maximum value for manual scaling (optional)
            obj: The node instance to pass to the on_change callback (optional)
        """
        old_config = {
            "scale_mode": self.scale_mode,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
        }
        self.scale_mode = scale_mode
        if scale_min is not None:
            self.scale_min = scale_min
        if scale_max is not None:
            self.scale_max = scale_max
        # Trigger on_change callback if configured
        if self.on_change and obj is not None:
            self._trigger_change(obj, {
                "scale_mode": self.scale_mode,
                "scale_min": self.scale_min,
                "scale_max": self.scale_max,
            }, old_config)

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "linechart",
            "label": self.label,
            "color": self.color,
            "line_width": self.line_width,
            "scale_mode": self.scale_mode,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "shape": list(value.shape) if value is not None and hasattr(value, "shape") else None,
        }


class Vector2D(Display):
    """2D array visualization descriptor."""
    def __init__(self, label: str, color_mode: str = "grayscale", on_change: Union[str, Callable, None] = None):
        super().__init__(label, on_change=on_change)
        self.color_mode = color_mode

    def change_color_mode(self, new_mode: str, obj: Any = None) -> None:
        """Change color mode at runtime and notify the UI.
        
        Args:
            new_mode: The new color mode (e.g., "grayscale", "bwr", "rgb")
            obj: The node instance to pass to the on_change callback (optional, for auto-notification)
        """
        old_mode = self.color_mode
        if new_mode != old_mode:
            self.color_mode = new_mode
            # Trigger on_change callback if configured
            if self.on_change and obj is not None:
                self._trigger_change(obj, {"color_mode": new_mode}, {"color_mode": old_mode})

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
