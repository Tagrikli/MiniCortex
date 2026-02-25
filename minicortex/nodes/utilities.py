import numpy as np
from typing import Optional, Dict, Any

from ..core.node import Node
from ..core.descriptors.ports import InputPort, OutputPort
from ..core.descriptors.properties import Slider, Integer
from ..core.descriptors.displays import Vector2D, Text
from ..core.descriptors.actions import Button
from ..core.descriptors.store import Store
from ..core.descriptors import dynamic, node


@dynamic
@node.utility
class AddNoise(Node):
    """Add Gaussian noise to input array."""
    
    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    mean = Slider("Mean", 0.0, -1.0, 1.0, scale="linear")
    std = Slider("Std", 0.1, 0.0, 1.0, scale="linear")
    result = Vector2D("Output", color_mode="bwr")
    info = Text("Info", default="Noise: μ=0.0, σ=0.1")
    
    
    def init(self):
        self._rng = np.random.default_rng()
        self.info = f"Noise: μ={self.mean:.2f}, σ={self.std:.2f}"

    def process(self):
        if self.input_data is None:
            return
        output = self.input_data + self._rng.normal(self.mean, self.std, size=self.input_data.shape)
        self.output_data = output
        if output.ndim == 2:
            self.output_data[0][0] = self.std
            self.result = output
        self.info = f"Noise: μ={self.mean:.2f}, σ={self.std:.2f}"


@node.utility
class Invert(Node):
    """Invert input array (1.0 - input)."""
    
    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    result = Vector2D("Output", color_mode="grayscale")
    info = Text("Info", default="Inverted")

    def process(self):
        if self.input_data is None:
            return
        output = 1.0 - self.input_data
        self.output_data = output
        self.result = output
        self.info = "Inverted"


class DisplayBase(Node):
    """Base class for display nodes."""
    
    input_data = InputPort("Input", np.ndarray)
    display = Vector2D("Display", color_mode="grayscale")

    def process(self):
        if self.input_data is not None and self.input_data.ndim == 2:
            self.display = self.input_data


@node.utility
class DisplayGrayscale(DisplayBase):
    """Display 2D array as grayscale image."""
    
    display = Vector2D("Display", color_mode="grayscale")


@node.utility
class DisplayBWR(DisplayBase):
    """Display 2D array with blue-white-red colormap."""
    
    display = Vector2D("Display", color_mode="bwr")


@node.utility
class AddArrays(Node):
    """Add two arrays together."""
    
    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)
    output_data = OutputPort("Result", np.ndarray)

    def process(self):
        if self.input_a is not None and self.input_b is not None:
            try:
                self.output_data = self.input_a + self.input_b
            except:
                self.output_data = None


@node.utility
class Duplicate(Node):
    """Pass through integer value."""
    
    input_data = InputPort("Input", int)
    output_data = OutputPort("Output", int)

    def init(self):
        self.output_data = 0

    def process(self):
        if self.input_data is not None:
            self.output_data = self.input_data


@node.utility
class AddIntegers(Node):
    """Add two integers together."""
    
    input_1 = InputPort("Input 1", int)
    input_2 = InputPort("Input 2", int)
    output_data = OutputPort("Result", int)
    result = Text("Result", default="0")

    def init(self):
        self.output_data = 0
        self.result = "0"

    def process(self):
        if self.input_1 is not None and self.input_2 is not None:
            self.output_data = int(self.input_1) + int(self.input_2)
            self.result = str(self.output_data)


@node.utility
class MovingAverage2D(Node):
    """Compute moving average of 2D arrays."""
    
    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Result", np.ndarray)
    size = Integer("Size", default=28, min_val=1)
    alpha = Slider("Alpha", 0.1, 0.0, 1.0, scale="linear")
    reinit = Button("Reinit", callback="_on_reinit")
    
    # Store the running average
    avg = Store(default=None)

    def init(self):
        if self.avg is None:
            self.avg = np.zeros((self.size, self.size), dtype=np.float32)

    def _on_reinit(self, params: dict):
        self.avg = np.zeros((self.size, self.size), dtype=np.float32)
        return {"status": "ok"}

    def process(self):
        if self.input_data is not None and isinstance(self.input_data, np.ndarray) and self.input_data.ndim == 2:
            h, w = self.input_data.shape
            if h == w and h != self.size:
                self.size = h
            if self.avg is None or self.avg.shape != self.input_data.shape:
                self.avg = self.input_data.astype(np.float32, copy=True)
            else:
                a = float(self.alpha)
                self.avg = (1.0 - a) * self.avg + a * self.input_data.astype(np.float32, copy=False)
            self.output_data = self.avg
