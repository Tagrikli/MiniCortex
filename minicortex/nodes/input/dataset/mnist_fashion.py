import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.displays import Vector2D, Text
from ....core.descriptors.store import Store
from ....core.descriptors import branch
from ....nodes.utilities import _load_dataset_with_python_mnist


class InputFashionMNIST(Node):
    """Cycle through Fashion-MNIST images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_category = OutputPort("Category", int)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    info = Text("Info", default="Fashion: 0")
    
    # Store: current image index
    idx = Store(default=0)
    
    LABEL_NAMES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    def init(self):
        self._images, self._labels = _load_dataset_with_python_mnist("fashion_mnist")
        self._update_pattern()

    def process(self):
        if self._images is not None:
            self.idx = (self.idx + 1) % len(self._images)
        self._update_pattern()

    def _update_pattern(self):
        if self._images is not None:
            self.output_pattern = self._images[self.idx]
            self.pattern = self._images[self.idx]
            l = int(self._labels[self.idx])
            self.output_category = l
            self.info = (
                f"Fashion: {self.LABEL_NAMES[l] if l < len(self.LABEL_NAMES) else l}"
            )
