from typing import Any, List, Optional
from .base import BaseDescriptor


class Action(BaseDescriptor):
    """Action descriptor for triggering callbacks (e.g., button clicks)."""
    
    def __init__(self, label: str, callback: str, params: Optional[List[dict]] = None, confirm: bool = False):
        super().__init__(label)
        self.callback = callback
        self.params = params or []
        self.confirm = confirm

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        callback = getattr(obj, self.callback, None)
        if callable(callback):
            return callback
        return lambda params=None: None

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "action",
            "label": self.label,
            "callback": self.callback,
            "params": self.params,
            "confirm": self.confirm,
        }
