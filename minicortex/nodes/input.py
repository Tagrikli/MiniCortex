from typing import Dict, Optional, Any
from pathlib import Path
import os
import numpy as np

from ..core.node import Node
from ..core.descriptors.ports import InputPort, OutputPort
from ..core.descriptors.properties import Slider, Integer
from ..core.descriptors.displays import Vector2D, Text
from ..core.descriptors.store import Store
from ..core.descriptors.node import node


def _load_dataset_with_python_mnist(dataset: str):
    from mnist import MNIST

    dataset_key = dataset.lower()
    env_var = (
        "MINICORTEX_MNIST_DIR"
        if dataset_key == "mnist"
        else "MINICORTEX_FASHION_MNIST_DIR"
    )
    candidates = []
    env_path = os.environ.get(env_var)
    if env_path:
        candidates.append(Path(env_path))
    if dataset_key == "mnist":
        candidates.extend(
            [
                Path("data/mnist/mnist"),
                Path("data/mnist"),
                Path("data/MNIST"),
                Path("datasets/mnist"),
                Path("mnist"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("data/mnist/fashion-mnist"),
                Path("data/mnist/fashion_mnist"),
                Path("data/fashion-mnist"),
                Path("data/fashion_mnist"),
                Path("data/FashionMNIST"),
                Path("datasets/fashion-mnist"),
                Path("fashion-mnist"),
            ]
        )

    last_error = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            mndata = MNIST(str(path))
            if any(path.glob("*.gz")):
                mndata.gz = True
            images, labels = mndata.load_training()
            images = np.asarray(images, dtype=np.float32).reshape((-1, 28, 28)) / 255.0
            return images, np.asarray(labels, dtype=np.int32)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load {dataset}. Set {env_var}.") from last_error


@node.input
class InputRotatingLine(Node):
    """Generate a rotating line pattern."""

    o_pattern = OutputPort("Pattern", np.ndarray)
    p_delta = Slider("Delta", 0.01, 0, 1)
    d_pattern = Vector2D("Pattern", color_mode="grayscale")
    d_angle = Text("Angle", default="0.00 rad")
    
    # Store for internal state
    angle = Store(default=0.0)
    size = Store(default=28)

    def init(self):
        self.process()

    def process(self):
        pattern = np.zeros((self.size, self.size))
        cx, cy = self.size // 2, self.size // 2
        length = self.size // 2 - 2
        for r in range(-length, length + 1):
            x, y = int(cx + r * np.cos(self.angle)), int(cy + r * np.sin(self.angle))
            if 0 <= x < self.size and 0 <= y < self.size:
                pattern[y, x] = 1.0
        self.o_pattern = self.d_pattern = pattern
        self.d_angle = f"{self.angle:.2f} rad"
        self.angle += float(self.p_delta)


@node.input
class InputScanningSquare(Node):
    """Generate a scanning square pattern."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    info = Text("Info", default="Position: 0")
    
    # Store for internal state (renamed to avoid conflict with Node.position)
    scan_pos = Store(default=0)
    size = Store(default=28)

    def init(self):
        self._update_pattern()

    def process(self):
        self.scan_pos = (self.scan_pos + 1) % (self.size * 2)
        self._update_pattern()

    def _update_pattern(self):
        pattern = np.zeros((self.size, self.size))
        sq_size = max(3, self.size // 4)
        pos = self.scan_pos % (self.size * 2)
        x = y = pos if pos < self.size else 2 * self.size - pos - 1
        half = sq_size // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px, py = x + dx, y + dy
                if 0 <= px < self.size and 0 <= py < self.size:
                    pattern[py, px] = 1.0
        self.output_pattern = pattern
        self.pattern = pattern
        self.info = f"Position: {self.scan_pos}"


@node.input
class InputDigitMNIST(Node):
    """Cycle through MNIST digit images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_digit = OutputPort("Digit", int)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    digit = Text("Digit", default="0")
    
    # Store only the index, images loaded from dataset (transient)
    idx = Store(default=0)
    size = Store(default=28)

    def init(self):
        self._images, self._labels = _load_dataset_with_python_mnist("mnist")
        self._update_pattern()

    def process(self):
        if self._images is not None:
            self.idx = (self.idx + 1) % len(self._images)
        self._update_pattern()

    def _update_pattern(self):
        if self._images is not None:
            self.output_pattern = self._images[self.idx]
            self.pattern = self._images[self.idx]
            self.output_digit = int(self._labels[self.idx])
            self.digit = str(self._labels[self.idx])


@node.input
class InputFashionMNIST(Node):
    """Cycle through Fashion-MNIST images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    info = Text("Info", default="Fashion: 0")
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
    
    # Store only the index, images loaded from dataset (transient)
    idx = Store(default=0)
    size = Store(default=28)

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
            self.info = (
                f"Fashion: {self.LABEL_NAMES[l] if l < len(self.LABEL_NAMES) else l}"
            )


@node.input
class InputInteger(Node):
    """Output an integer value."""

    output_value = OutputPort("Value", int)
    value = Integer("Value", default=0)

    def init(self):
        self.output_value = 0

    def process(self):
        self.output_value = int(self.value)

