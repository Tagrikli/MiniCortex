"""Utility nodes for MiniCortex."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Integer
from axonforge.core.descriptors.displays import Text, Numeric


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
        p = v**2

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




class TopK(Node):
    """Keep only the top K values in an array, set rest to zero."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    k = Integer("K", default=10)

    def process(self):
        if self.input_data is None:
            return

        # Store original shape
        original_shape = self.input_data.shape

        # Flatten the array
        flat = self.input_data.flatten().astype(np.float64)

        # Get k value (ensure it doesn't exceed array size)
        k_val = min(int(self.k), len(flat))

        if k_val <= 0:
            self.output_data = np.zeros(original_shape, dtype=np.float32)
            return

        # Find the k-th largest value (threshold)
        # Use partition for efficiency: all values >= threshold are in the top k
        threshold = np.partition(flat, -k_val)[-k_val]

        # Create output: keep top k values, set rest to zero
        result = np.where(flat >= threshold, flat, 0.0)

        # Reshape back to original shape
        self.output_data = result.reshape(original_shape).astype(np.float32)


class JaccardIndex(Node):
    """Calculate Jaccard Index (Intersection over Union) between two arrays.

    Converts positive values to 1 and non-positive values to 0.
    Then computes: intersection / union
    """

    input_true = InputPort("True", np.ndarray)
    input_predicted = InputPort("Predicted", np.ndarray)
    output_iou = OutputPort("IoU", float)

    display = Numeric("Jaccard Index", format=".4f")
    info = Text("Info", default="No data")

    def process(self):
        if self.input_true is None or self.input_predicted is None:
            self.info = "Waiting for inputs"
            return

        # Flatten arrays
        a = self.input_true.flatten().astype(np.float32)
        b = self.input_predicted.flatten().astype(np.float32)

        # Check if arrays are same size
        if len(a) != len(b):
            self.info = f"Size mismatch: {len(a)} vs {len(b)}"
            self.output_iou = 0.0
            self.display = 0.0
            return

        # Convert to binary: positive values -> 1, non-positive -> 0
        a_binary = (a > 0).astype(np.int32)
        b_binary = (b > 0).astype(np.int32)

        # Calculate intersection and union
        intersection = np.sum(a_binary & b_binary)
        union = np.sum(a_binary | b_binary)

        # Calculate Jaccard Index
        jaccard = intersection / union if union > 0 else 0.0

        # Set outputs
        self.output_iou = float(jaccard)
        self.display = float(jaccard)
        self.info = f"Jaccard Index: {jaccard:.4f}\nIntersection: {intersection}\nUnion: {union}"
