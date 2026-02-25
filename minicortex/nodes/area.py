import numpy as np
from ..core.node import Node
from ..core.descriptors.ports import InputPort, OutputPort
from ..core.descriptors.properties import Slider, Integer
from ..core.descriptors.displays import Numeric, Vector1D, Vector2D
from ..core.descriptors.store import Store
from ..core.descriptors import node


@node.processing
class Area(Node):
    """
    Area node representing a cortical processing layer.
    
    Note: Weights are not stored via Store() because they can be very large
    (num_minicolumn * input_size^2 floats). They will be reinitialized on load.
    If persistence is needed later, consider adding conditional storage with
    a size limit or separate file-based storage.
    """
    
    # Ports
    input_data = InputPort("Input Stream", np.ndarray)
    output_data = OutputPort("Output Stream", np.ndarray)

    # Properties
    alpha = Slider("Learning Rate (α)", 0.01, 1e-5, 1.0, scale="log")
    beta = Slider("Inhibition (β)", 0.5, 0.0, 2.0)
    gamma = Slider("Weight Decay (γ)", 0.0001, 0.0, 0.1)
    lam = Slider("Trace Decay (λ)", 0.9, 0.0, 1.0)
    
    input_size = Integer("Input Dim", 28, min_val=1)
    num_minicolumn = Integer("Minicolumns", 9, min_val=1)

    # Display Outputs
    weights_viz = Vector2D("Weights Viz")
    similarity_out = Vector1D("Similarities")
    activation_out = Vector1D("Activations")
    loss_out = Numeric("Loss")



    def init(self):
        """Initialize weights after properties are set."""
        print(f"Initializing Area {self.name} with {self.num_minicolumn} columns")
        # Initialize weights: (num_minicolumn, input_size * input_size)
        self._weights = np.random.rand(int(self.num_minicolumn), int(self.input_size) * int(self.input_size))
        self.weights = self._weights  # Sync display
        self.weights_viz = self._weights.reshape(3, 3, 28, 28)  # Hardcoded reshape for MNIST for now

    def process(self):
        """
        Process input pattern through competitive learning.
        """
        if self.input_data is None:
            return

        # Simple placeholder for competitive learning logic
        # 1. Calculate similarity between input and each minicolumn weight vector
        A_flat = self._weights.reshape(int(self.num_minicolumn), -1)
        v_flat = self.input_data.reshape(-1)

        # Ensure shapes match
        if A_flat.shape[1] != v_flat.shape[0]:
            # This can happen if input_size doesn't match connected input
            return

        self.similarities = A_flat @ v_flat
        
        # 2. Competitive inhibition (WTA or Softmax)
        # Placeholder: Softmax
        exp_sim = np.exp(self.similarities * 10 / (np.max(self.similarities) + 1e-9))
        self.activations = exp_sim / np.sum(exp_sim)
        
        # 3. Learning (Hebb-like update toward current input)
        winner_idx = np.argmax(self.activations)
        self._weights[winner_idx] += float(self.alpha) * (v_flat - self._weights[winner_idx])
        
        # 4. Decay
        self._weights *= (1.0 - float(self.gamma))
        
        # Update display outputs
        self.similarity_out = self.similarities
        self.activation_out = self.activations
        self.weights_viz = self._weights.reshape(3, 3, 28, 28)  # Hardcoded reshape
        self.loss_out = np.linalg.norm(self._weights[winner_idx] - v_flat)
        
        # Set output
        self.output_data = self.activations
