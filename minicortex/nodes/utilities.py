import numpy as np
from typing import Optional, Dict, Any

from ..core.node import Node
from ..core.descriptors.ports import InputPort, OutputPort
from ..core.descriptors.properties import Range, Integer
from ..core.descriptors.displays import Vector2D, Vector1D, Text, Numeric
from ..core.descriptors.actions import Action
from ..core.descriptors.store import Store
from ..core.descriptors import dynamic, node


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
    alpha = Range("Alpha", 0.1, 0.0, 1.0, scale="linear")
    reinit = Action("Reinit", callback="_on_reinit")
    
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


@node.utility
class L2Normalize(Node):
    """L2 normalize a numpy array. Flattens if required, then reshapes to original shape."""
    
    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)

    def process(self):
        if self.input_data is None:
            return
        
        # Store original shape
        original_shape = self.input_data.shape
        
        # Flatten the array for L2 normalization
        flat = self.input_data.flatten().astype(np.float64)
        
        # Compute L2 norm
        l2_norm = np.linalg.norm(flat)
        
        # Normalize (avoid division by zero)
        if l2_norm > 0:
            normalized = flat / l2_norm
        else:
            normalized = flat
        
        # Reshape back to original shape
        self.output_data = normalized.reshape(original_shape).astype(np.float32)


@node.utility
class Display1D(Node):
    """Display 1D numpy array as a bar chart with optional normalization."""
    
    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    display = Vector1D("Plot")
    min_val = Range("Min", 0.0, -1.0, 1.0, scale="linear")
    max_val = Range("Max", 1.0, -1.0, 1.0, scale="linear")
    info = Text("Info", default="No data")

    def process(self):
        if self.input_data is None:
            self.info = "No data"
            return
        
        # Flatten to 1D
        arr = self.input_data.flatten().astype(np.float64)
        
        # Get normalization bounds
        min_bound = float(self.min_val)
        max_bound = float(self.max_val)
        
        # Normalize array to [min_bound, max_bound] range
        arr_min = arr.min()
        arr_max = arr.max()
        
        if arr_max - arr_min > 1e-10:
            # Normalize to [0, 1] first, then scale to target range
            normalized = (arr - arr_min) / (arr_max - arr_min)
            normalized = min_bound + normalized * (max_bound - min_bound)
        else:
            # Constant array - set to midpoint
            normalized = np.full_like(arr, (min_bound + max_bound) / 2.0)
        
        # Set outputs
        self.output_data = normalized.astype(np.float32)
        self.display = normalized.astype(np.float32)
        self.info = f"Shape: {self.input_data.shape} → {normalized.shape[0]}"


@node.utility
class FloatHistory(Node):
    """Track history of float values and display as a plot."""
    
    input_value = InputPort("Value", float)
    output_data = OutputPort("History", np.ndarray)
    history_size = Integer("History Size", default=10, min_val=2, max_val=1000)
    display = Vector1D("Plot")
    info = Text("Info", default="No data")
    
    # Store for history buffer
    history = Store(default=None)

    def init(self):
        if self.history is None:
            self.history = np.zeros(self.history_size, dtype=np.float32)

    def process(self):
        if self.input_value is None:
            return
        
        # Initialize history if needed
        if self.history is None or len(self.history) != int(self.history_size):
            self.history = np.zeros(int(self.history_size), dtype=np.float32)
        
        # Shift history left and add new value at the end
        self.history = np.roll(self.history, -1)
        self.history[-1] = float(self.input_value)
        
        # Output and display
        self.output_data = self.history.copy()
        self.display = self.history.copy()
        self.info = f"Last: {float(self.input_value):.4f}"


@node.utility
class Uniformity(Node):
    """Calculate non-uniformity of a numpy array using entropy-based measure."""
    
    input_data = InputPort("Input", np.ndarray)
    non_uniformity = OutputPort("Non-Uniformity", float)
    entropy = OutputPort("Entropy", float)
    display = Numeric("Non-Uniformity", format=".4f")
    info = Text("Info", default="No data")

    def process(self):
        if self.input_data is None:
            return
        
        # Flatten and L2 normalize
        v = self.input_data.flatten().astype(np.float64)
        norm = np.linalg.norm(v)
        
        if norm > 0:
            v = v / norm
        else:
            self.non_uniformity = 0.0
            self.entropy = 0.0
            self.display = 0.0
            self.info = "Zero vector"
            return
        
        # Compute probability distribution (sums to 1 by construction)
        p = v ** 2
        
        # Compute entropy
        entropy_val = -np.sum(p * np.log(p + 1e-9))
        
        # Compute non-uniformity (0 = uniform, 1 = concentrated)
        max_entropy = np.log(v.size)
        non_uniformity_val = 1.0 - entropy_val / max_entropy if max_entropy > 0 else 0.0
        
        # Set outputs
        self.non_uniformity = float(non_uniformity_val)
        self.entropy = float(entropy_val)
        self.display = float(non_uniformity_val)
        uniformity_val = 1.0 - non_uniformity_val
        self.info = f"Uniformity: {uniformity_val:.4f}\nNon-Uniformity: {non_uniformity_val:.4f}\nEntropy: {entropy_val:.4f}\nMax Entropy: {max_entropy:.4f}"
