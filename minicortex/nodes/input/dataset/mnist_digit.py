"""Dataset input nodes for MiniCortex."""


import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.properties import Integer
from ....core.descriptors.displays import Vector2D, Text
from ....core.descriptors.store import Store
from ....core.descriptors import branch
from ....nodes.utilities import _load_dataset_with_python_mnist



class InputDigitMNIST(Node):
    """Cycle through MNIST digit images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_digit = OutputPort("Digit", int)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    digit = Text("Digit", default="0")
    
    # Property: how many times to repeat each image
    repeat_count = Integer("Repeat", default=1)
    
    # Store: index and step counter
    idx = Store(default=0)
    repeat_counter = Store(default=0)
    size = Store(default=28)

    def init(self):
        self._images, self._labels = _load_dataset_with_python_mnist("mnist")
        self._update_pattern()

    def process(self):
        if self._images is not None:
            # Increment repeat counter
            self.repeat_counter = int(self.repeat_counter) + 1
            
            # Only advance to next digit when repeat_count is reached
            if int(self.repeat_counter) >= int(self.repeat_count):
                self.idx = (int(self.idx) + 1) % len(self._images)
                self.repeat_counter = 0
        
        self._update_pattern()

    def _update_pattern(self):
        if self._images is not None:
            self.output_pattern = self._images[self.idx]
            self.pattern = self._images[self.idx]
            self.output_digit = int(self._labels[self.idx])
            self.digit = str(self._labels[self.idx])



