from typing import Any, List, Optional
from .base import Action

class Button(Action):
    """Action button descriptor."""
    def __init__(self, label: str, callback: str, params: Optional[List[dict]] = None, confirm: bool = False):
        super().__init__(label, callback)
        self.params = params or []
        self.confirm = confirm

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "button",
            "label": self.label,
            "callback": self.callback,
            "params": self.params,
            "confirm": self.confirm,
        }
