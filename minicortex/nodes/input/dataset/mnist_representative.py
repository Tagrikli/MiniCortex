"""MNIST Representative image node for MiniCortex.

This node takes an integer digit (0-9) as input and outputs a representative
image for that digit category from the MNIST dataset. The representative image
is computed as the mean (average) of all images in the training set for each
digit class, providing a prototypical representation.

Additionally, the node can compute activations by performing a dot product
between the flattened representative image and an input weight matrix.
"""

import numpy as np

from ....core.node import Node
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.displays import Vector2D, Text
from ....core.descriptors.store import Store
from ....nodes.utilities import _load_dataset_with_python_mnist


class MNISTRepresentative(Node):
    """Output a representative MNIST image for a given digit with activation computation.
    
    This node loads the MNIST dataset and computes representative (mean/average)
    images for each digit category (0-9). When given an input digit, it outputs
    the pre-computed representative image for that digit.
    
    The representative image is calculated as the mean of all training images
    for each digit class, providing a prototypical representation that captures
    the average characteristics of each digit.
    
    If weights are provided, the node computes activations by flattening the
    representative image and performing a dot product with the weights matrix.
    
    Inputs:
        i_digit: Integer digit (0-9) to get the representative image for
        i_weights: Weight matrix for computing activations (2D array, shape [784, N])
        
    Outputs:
        o_image: 28x28 numpy array containing the representative image
        o_activations: 1D array of activations (computed as flattened_image · weights)
        
    Displays:
        preview: Visual display of the representative image
        info: Text info showing current digit, image stats, and activation info
    """

    # ── Ports ─────────────────────────────────────────────────────────────
    i_digit = InputPort("Digit", int)
    i_weights = InputPort("Weights", np.ndarray)
    o_image = OutputPort("Image", np.ndarray)
    o_activations = OutputPort("Activations", np.ndarray)

    # ── Displays ───────────────────────────────────────────────────────────
    preview = Vector2D("Preview", color_mode="grayscale")
    info = Text("Info", default="No input")

    # ── Store ─────────────────────────────────────────────────────────────
    current_digit = Store(default=None)
    size = Store(default=28)

    def init(self):
        """Initialize the node by loading MNIST and computing representative images."""
        self._representative_images = {}
        self._load_and_compute_representatives()
        
    def _load_and_compute_representatives(self):
        """Load MNIST dataset and compute representative images for each digit."""
        images, labels = _load_dataset_with_python_mnist("mnist")
        
        if images is None or labels is None:
            self.info = "Error: Could not load MNIST dataset"
            return
        
        # Compute representative (mean) image for each digit class
        for digit in range(10):
            # Get all images for this digit
            mask = labels == digit
            digit_images = images[mask]
            
            if len(digit_images) > 0:
                # Compute mean image (average of all images for this digit)
                mean_image = np.mean(digit_images, axis=0)
                self._representative_images[digit] = mean_image
            else:
                # Fallback: empty image if no data (shouldn't happen with MNIST)
                self._representative_images[digit] = np.zeros((28, 28), dtype=np.float32)
        
        self.info = f"Loaded {len(images)} images, computed 10 representatives"

    def process(self):
        """Process the input digit and output the representative image and activations."""
        if self.i_digit is None:
            return
        
        # Get and validate the input digit
        digit = int(self.i_digit)
        digit = max(0, min(9, digit))  # Clamp to valid range
        
        # Get the representative image for this digit
        if digit not in self._representative_images:
            self.info = f"Error: No representative for digit {digit}"
            return
        
        rep_image = self._representative_images[digit]
        
        # Set image outputs
        self.o_image = rep_image
        self.preview = rep_image
        self.current_digit = digit
        
        # Compute activations if weights are provided
        activations_info = ""
        if self.i_weights is not None:
            # Flatten the image (28x28 -> 784)
            flat_image = rep_image.flatten()
            
            # Compute activations using transposed dot product
            # If weights shape is (n_neurons, 784), we transpose to (784, n_neurons)
            # Result: (784,) · (784, n_neurons) = (n_neurons,)
            weights = self.i_weights
            if weights.ndim == 2 and weights.shape[1] == flat_image.shape[0]:
                # weights is (n_neurons, 784), transpose to (784, n_neurons)
                activations = np.dot(flat_image, weights.T)
            else:
                # Fallback: try direct dot product
                activations = np.dot(flat_image, weights)
            self.o_activations = activations
            
            # Activation statistics for info display
            act_min = float(np.min(activations))
            act_max = float(np.max(activations))
            act_mean = float(np.mean(activations))
            activations_info = f", Act: {activations.shape}, Min: {act_min:.3f}, Max: {act_max:.3f}, Mean: {act_mean:.3f}"
        else:
            self.o_activations = None
        
        # Update info display
        min_val = float(np.min(rep_image))
        max_val = float(np.max(rep_image))
        mean_val = float(np.mean(rep_image))
        self.info = f"Digit: {digit}, Img: [{min_val:.3f}, {max_val:.3f}, {mean_val:.3f}]{activations_info}"
