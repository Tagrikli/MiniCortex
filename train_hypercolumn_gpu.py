#!/usr/bin/env python3
"""
GPU-accelerated HyperColumn training script using CuPY.

Replicates the HyperColumnFieldDriven.process() function for GPU calculations.
Trains on MNIST dataset with class-specific feedback vectors.
Accuracy metric: cosine similarity between output activations and injected feedback.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

# Try to import CuPY for GPU acceleration, fallback to NumPy if unavailable
try:
    import cupy as cp
    from cupy.cuda import Device
    GPU_AVAILABLE = True
    print("CuPY imported successfully - GPU acceleration enabled")
    # Print GPU info
    try:
        device = Device(0)
        print(f"  Using GPU: {device.mem_info}")
    except:
        print("  GPU device info unavailable")
except ImportError:
    print("CuPY not available, falling back to NumPy (CPU mode)")
    cp = np  # Fallback to numpy
    GPU_AVAILABLE = False

# ==============================================================================
# GLOBAL PARAMETERS (from HyperColumnFieldDriven class defaults)
# ==============================================================================
INPUT_LEN = 28                    # Input image size (28x28)
MINICOLUMN_COUNT = 1024*4             # Number of minicolumns
ALPHA = 0.01                       # Learning rate (α)
BETA = 0.0001                      # Inhibition parameter (β)
FEEDBACK_RATIO = 0.1              # Feedback ratio (ρ)
EPS = 1e-8                        # Small constant for numerical stability
NUM_CLASSES = 10                  # MNIST has 10 classes (0-9)
BATCH_SIZE = 32                   # Training batch size
NUM_EPOCHS = 5                    # Number of training epochs

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_mnist_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset using python-mnist.
    
    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels)
    """
    try:
        from mnist import MNIST
    except ImportError:
        raise ImportError(
            "python-mnist not installed. Install with: pip install python-mnist"
        )
    
    # Determine data directory
    data_path = Path(__file__).parent / "data" / "mnist" / "mnist"
    
    if not data_path.exists():
        # Try legacy path
        data_path = Path(__file__).parent / "data" / "mnist_data"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"MNIST dataset not found at {data_path}. "
            "Run utils/download_mnist_datasets.py first."
        )
    
    # Load dataset
    mnist_data = MNIST(str(data_path))
    train_images, train_labels = mnist_data.load_training()
    test_images, test_labels = mnist_data.load_testing()
    
    # Convert to numpy arrays and normalize
    train_images = np.array(train_images).reshape(-1, 28, 28).astype(np.float32) / 255.0
    train_labels = np.array(train_labels)
    test_images = np.array(test_images).reshape(-1, 28, 28).astype(np.float32) / 255.0
    test_labels = np.array(test_labels)
    
    print(f"Loaded MNIST dataset:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    return train_images, train_labels, test_images, test_labels


def create_class_feedback_vectors(vector_size: int, num_classes: int, seed: int = 42) -> np.ndarray:
    """
    Create random unit vectors for each class to use as feedback.
    
    Args:
        vector_size: Dimension of each feedback vector (minicolumn_count)
        num_classes: Number of classes (10 for MNIST)
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (num_classes, vector_size) with unit vectors
    """
    rng = np.random.RandomState(seed)
    feedback_vectors = rng.randn(num_classes, vector_size).astype(np.float32)
    
    # Normalize to unit length
    norms = np.linalg.norm(feedback_vectors, axis=1, keepdims=True) + EPS
    feedback_vectors = feedback_vectors / norms
    
    return feedback_vectors


# ==============================================================================
# GPU-ACCELERATED HYPERCOLUMN PROCESS FUNCTION
# ==============================================================================

class HyperColumnGPU:
    """
    GPU-accelerated HyperColumn implementation.
    Replicates the HyperColumnFieldDriven.process() function.
    """
    
    def __init__(
        self,
        input_len: int = INPUT_LEN,
        minicolumn_count: int = MINICOLUMN_COUNT,
        alpha: float = ALPHA,
        beta: float = BETA,
        feedback_ratio: float = FEEDBACK_RATIO,
        is_learning: bool = True
    ):
        """
        Initialize the HyperColumn.
        
        Args:
            input_len: Input image size (default 28 for MNIST)
            minicolumn_count: Number of minicolumns
            alpha: Learning rate
            beta: Inhibition parameter
            feedback_ratio: Ratio for mixing feedback (ρ)
            is_learning: Whether to update weights during processing
        """
        self.input_len = input_len
        self.minicolumn_count = minicolumn_count
        self.alpha = alpha
        self.beta = beta
        self.feedback_ratio = feedback_ratio
        self.is_learning = is_learning
        
        # Initialize weights on GPU
        self.weights = self._initialize_weights()
        
        # Initialize maps (on GPU)
        self.map_pix = cp.zeros(self.input_len ** 2, dtype=cp.float32)
        self.map_rep = cp.zeros(self.minicolumn_count, dtype=cp.float32)
        
        # Beta per neuron (adaptive inhibition)
        self.beta_per_neuron = cp.full(self.minicolumn_count, self.beta, dtype=cp.float32)
    
    def _initialize_weights(self) -> cp.ndarray:
        """Initialize random unit vectors as weights on GPU."""
        weights = cp.random.randn(self.minicolumn_count, self.input_len ** 2).astype(cp.float32)
        norms = cp.linalg.norm(weights, axis=1, keepdims=True) + EPS
        weights = weights / norms
        return weights
    
    def process(
        self,
        x_input: cp.ndarray,
        feedback: Optional[cp.ndarray] = None,
        return_reconstruction_error: bool = False
    ) -> cp.ndarray | Tuple[cp.ndarray, float]:
        """
        Process a single input through the HyperColumn.
        
        Replicates the HyperColumnFieldDriven.process() function.
        
        Args:
            x_input: Input array, shape (input_len, input_len) or (input_len**2,)
            feedback: Optional feedback vector, shape (minicolumn_count,)
            return_reconstruction_error: If True, also return reconstruction MSE
            
        Returns:
            Output activations s_final, shape (minicolumn_count,)
            If return_reconstruction_error is True, returns (output, reconstruction_mse)
        """
        eps = EPS
        rho = self.feedback_ratio
        
        ######## INPUT ########
        # Flatten input if needed
        if x_input.ndim > 1:
            x = x_input.flatten()
        else:
            x = x_input
        
        # Normalize input
        x_norm_factor = cp.linalg.norm(x) + eps
        x_norm = x / x_norm_factor
        
        ######## FEEDFORWARD ########
        # Compute raw activations: x_norm @ weights.T
        s_raw = x_norm @ self.weights.T
        
        # Apply gating/inhibition
        gate = 1.0 - self.beta * self.beta_per_neuron
        s_final = s_raw * gate
        
        ######## RECONSTRUCTION ########
        # Reconstruct input from activations
        x_hat = self.weights.T @ s_final
        
        ######## ERROR TERMS ########
        # --- World consistency (pixel error) ---
        e_pix = x_norm - x_hat
        
        # --- Hierarchical consistency (feedback error) ---
        if feedback is not None:
            # Normalize s_final and feedback for direction comparison
            s_hat = s_final / (cp.linalg.norm(s_final) + eps)
            f_hat = feedback / (cp.linalg.norm(feedback) + eps)
            
            delta = f_hat - s_hat
        else:
            delta = cp.zeros_like(s_final)
        
        ######## STORE MAPS ########
        self.map_pix = e_pix
        self.map_rep = delta
        
        ######## MIX ROTATIONS (TANGENT SPACE) ########
        # Pixel space tangent for reconstruction
        e_pix_hat = self.map_pix / (cp.linalg.norm(self.map_pix) + eps)
        dots_pix = self.weights @ e_pix_hat
        t_pix = e_pix_hat - dots_pix[:, None] * self.weights
        t_pix_hat = t_pix / (cp.linalg.norm(t_pix, axis=1, keepdims=True) + eps)
        
        # Activation space tangent for feedback
        dots_x = self.weights @ x_norm
        t_fb = x_norm - dots_x[:, None] * self.weights
        t_fb_hat = t_fb / (cp.linalg.norm(t_fb, axis=1, keepdims=True) + eps)
        
        # Feedback magnitude from activation mismatch
        e_rep_mag = cp.linalg.norm(delta)
        
        # Mix tangents
        t_mix = (1.0 - rho) * t_pix_hat + rho * delta[:, None] * t_fb_hat
        t_mix_hat = t_mix / (cp.linalg.norm(t_mix, axis=1, keepdims=True) + eps)
        
        ######## ADAPT INHIBITION ########
        a_pix = cp.clip(dots_pix, -1.0, 1.0)
        a_fb = cp.clip(delta, -1.0, 1.0)
        a = (1.0 - rho) * a_pix + rho * a_fb
        
        ######## LEARNING (ROTATIONAL UPDATE) ########
        if self.is_learning:
            # Update beta per neuron
            self.beta_per_neuron = 0.5 * (1.0 - a)
            
            # Compute rotation angles
            theta = self.alpha * self.beta_per_neuron * cp.maximum(s_final, 0)
            
            # Apply rotational update to weights
            self.weights = (
                cp.cos(theta)[:, None] * self.weights
                + cp.sin(theta)[:, None] * t_mix_hat
            )
            
            # Renormalize weights
            self.weights /= cp.linalg.norm(self.weights, axis=1, keepdims=True) + eps
        
        ######## OUTPUT ########
        # Scale output by input norm factor
        output = s_final * x_norm_factor
        
        # Calculate reconstruction error (MSE between x and x_hat)
        if return_reconstruction_error:
            # x_hat is the reconstruction in normalized space, compare to x_norm
            recon_error = cp.mean((x_norm - x_hat) ** 2)
            return output, float(recon_error)
        
        return output
    
    def process_batch(
        self,
        x_batch: cp.ndarray,
        feedback_batch: cp.ndarray,
        return_reconstruction_errors: bool = False
    ) -> cp.ndarray | Tuple[cp.ndarray, cp.ndarray]:
        """
        Process a batch of inputs.
        
        Args:
            x_batch: Batch of inputs, shape (batch_size, input_len**2)
            feedback_batch: Batch of feedback vectors, shape (batch_size, minicolumn_count)
            return_reconstruction_errors: If True, also return reconstruction MSE for each sample
            
        Returns:
            Batch of outputs, shape (batch_size, minicolumn_count)
            If return_reconstruction_errors is True, returns (outputs, reconstruction_errors)
        """
        batch_size = x_batch.shape[0]
        outputs = cp.zeros((batch_size, self.minicolumn_count), dtype=cp.float32)
        
        if return_reconstruction_errors:
            recon_errors = cp.zeros(batch_size, dtype=cp.float32)
            for i in range(batch_size):
                output, error = self.process(
                    x_batch[i], 
                    feedback_batch[i], 
                    return_reconstruction_error=True
                )
                outputs[i] = output
                recon_errors[i] = error
            return outputs, recon_errors
        else:
            for i in range(batch_size):
                outputs[i] = self.process(x_batch[i], feedback_batch[i])
            return outputs


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def calculate_cosine_similarity(
    outputs: cp.ndarray,
    feedbacks: cp.ndarray
) -> float:
    """
    Calculate mean cosine similarity between outputs and feedbacks.
    
    Args:
        outputs: Output activations, shape (batch_size, minicolumn_count)
        feedbacks: Feedback vectors, shape (batch_size, minicolumn_count)
        
    Returns:
        Mean cosine similarity across the batch
    """
    # Calculate norms
    output_norms = cp.linalg.norm(outputs, axis=1, keepdims=True) + EPS
    feedback_norms = cp.linalg.norm(feedbacks, axis=1, keepdims=True) + EPS
    
    # Normalize
    outputs_norm = outputs / output_norms
    feedbacks_norm = feedbacks / feedback_norms
    
    # Cosine similarity = dot product of normalized vectors
    similarities = cp.sum(outputs_norm * feedbacks_norm, axis=1)
    
    return float(cp.mean(similarities))


def calculate_classification_accuracy(
    outputs: cp.ndarray,
    labels: np.ndarray,
    feedback_vectors: np.ndarray
) -> float:
    """
    Calculate classification accuracy by comparing outputs to all class feedback vectors.
    
    Args:
        outputs: Output activations, shape (batch_size, minicolumn_count)
        labels: True class labels, shape (batch_size,)
        feedback_vectors: All class feedback vectors, shape (num_classes, minicolumn_count)
        
    Returns:
        Classification accuracy (0.0 to 1.0)
    """
    batch_size = outputs.shape[0]
    
    # Normalize outputs
    output_norms = cp.linalg.norm(outputs, axis=1, keepdims=True) + EPS
    outputs_norm = outputs / output_norms
    
    # Normalize all feedback vectors
    feedback_vectors_gpu = cp.asarray(feedback_vectors)
    feedback_norms = cp.linalg.norm(feedback_vectors_gpu, axis=1, keepdims=True) + EPS
    feedbacks_norm = feedback_vectors_gpu / feedback_norms
    
    # Calculate cosine similarity to all classes: (batch_size, num_classes)
    # outputs_norm @ feedbacks_norm.T -> (batch_size, num_classes)
    similarities = outputs_norm @ feedbacks_norm.T
    
    # Get predicted class (highest similarity)
    predictions = cp.argmax(similarities, axis=1)
    
    # Compare with true labels
    labels_gpu = cp.asarray(labels)
    correct = cp.sum(predictions == labels_gpu)
    
    return float(correct / batch_size)


def train_epoch(
    model: HyperColumnGPU,
    images: np.ndarray,
    labels: np.ndarray,
    feedback_vectors: np.ndarray,
    batch_size: int,
    epoch: int
) -> Tuple[float, float, float]:
    """
    Train for one epoch.
    
    Args:
        model: HyperColumn model
        images: Training images
        labels: Training labels
        feedback_vectors: Class feedback vectors
        batch_size: Batch size
        epoch: Current epoch number
        
    Returns:
        Tuple of (mean_recon_error, mean_similarity, classification_accuracy)
    """
    num_samples = len(images)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_similarity = 0.0
    total_recon_error = 0.0
    total_correct = 0
    
    # Create progress bar
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
    
    for batch_idx in pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        # Get batch data
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Flatten images
        batch_images_flat = batch_images.reshape(current_batch_size, -1)
        
        # Get feedback vectors for each sample based on its label
        batch_feedbacks = feedback_vectors[batch_labels]
        
        # Move to GPU
        batch_images_gpu = cp.asarray(batch_images_flat)
        batch_feedbacks_gpu = cp.asarray(batch_feedbacks)
        
        # Process batch with reconstruction errors
        outputs, recon_errors = model.process_batch(
            batch_images_gpu, 
            batch_feedbacks_gpu,
            return_reconstruction_errors=True
        )
        
        # Calculate cosine similarity (our accuracy metric)
        similarity = calculate_cosine_similarity(outputs, batch_feedbacks_gpu)
        total_similarity += similarity * current_batch_size
        
        # Calculate mean reconstruction error for this batch
        batch_recon_error = float(cp.mean(recon_errors))
        total_recon_error += batch_recon_error * current_batch_size
        
        # Calculate classification accuracy for this batch
        batch_accuracy = calculate_classification_accuracy(
            outputs, batch_labels, feedback_vectors
        )
        total_correct += batch_accuracy * current_batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'cos_sim': f'{similarity:.4f}',
            'recon_err': f'{batch_recon_error:.6f}',
            'acc': f'{batch_accuracy:.2%}'
        })
    
    mean_similarity = total_similarity / num_samples
    mean_recon_error = total_recon_error / num_samples
    classification_accuracy = total_correct / num_samples
    
    return mean_recon_error, mean_similarity, classification_accuracy


def evaluate(
    model: HyperColumnGPU,
    images: np.ndarray,
    labels: np.ndarray,
    feedback_vectors: np.ndarray,
    batch_size: int
) -> Tuple[float, float, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: HyperColumn model
        images: Test images
        labels: Test labels
        feedback_vectors: Class feedback vectors
        batch_size: Batch size
        
    Returns:
        Tuple of (mean_recon_error, mean_cosine_similarity, classification_accuracy)
    """
    num_samples = len(images)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_similarity = 0.0
    total_recon_error = 0.0
    total_correct = 0
    
    # Disable learning for evaluation
    original_learning = model.is_learning
    model.is_learning = False
    
    pbar = tqdm(range(num_batches), desc="Evaluating")
    
    for batch_idx in pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        # Get batch data
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Flatten images
        batch_images_flat = batch_images.reshape(current_batch_size, -1)
        
        # Get feedback vectors
        batch_feedbacks = feedback_vectors[batch_labels]
        
        # Move to GPU
        batch_images_gpu = cp.asarray(batch_images_flat)
        batch_feedbacks_gpu = cp.asarray(batch_feedbacks)
        
        # Process batch with reconstruction errors
        outputs, recon_errors = model.process_batch(
            batch_images_gpu, 
            batch_feedbacks_gpu,
            return_reconstruction_errors=True
        )
        
        # Calculate cosine similarity
        similarity = calculate_cosine_similarity(outputs, batch_feedbacks_gpu)
        total_similarity += similarity * current_batch_size
        
        # Calculate mean reconstruction error for this batch
        batch_recon_error = float(cp.mean(recon_errors))
        total_recon_error += batch_recon_error * current_batch_size
        
        # Calculate classification accuracy for this batch
        batch_accuracy = calculate_classification_accuracy(
            outputs, batch_labels, feedback_vectors
        )
        total_correct += batch_accuracy * current_batch_size
        
        pbar.set_postfix({
            'cos_sim': f'{similarity:.4f}',
            'recon_err': f'{batch_recon_error:.6f}',
            'acc': f'{batch_accuracy:.2%}'
        })
    
    # Restore learning state
    model.is_learning = original_learning
    
    mean_similarity = total_similarity / num_samples
    mean_recon_error = total_recon_error / num_samples
    classification_accuracy = total_correct / num_samples
    
    return mean_recon_error, mean_similarity, classification_accuracy


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main training script."""
    print("=" * 60)
    print("HyperColumn GPU Training")
    print("=" * 60)
    print()
    
    # Print configuration
    print("Configuration:")
    print(f"  GPU Mode: {'Enabled' if GPU_AVAILABLE else 'CPU Fallback'}")
    print(f"  Input Size: {INPUT_LEN}x{INPUT_LEN} = {INPUT_LEN**2}")
    print(f"  Minicolumns: {MINICOLUMN_COUNT}")
    print(f"  Alpha (learning rate): {ALPHA}")
    print(f"  Beta (inhibition): {BETA}")
    print(f"  Feedback Ratio (ρ): {FEEDBACK_RATIO}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print()
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_mnist_dataset()
    print()
    
    # Create class feedback vectors (random unit vectors for each digit class)
    print("Creating class feedback vectors...")
    feedback_vectors = create_class_feedback_vectors(
        vector_size=MINICOLUMN_COUNT,
        num_classes=NUM_CLASSES,
        seed=42
    )
    print(f"  Feedback vectors shape: {feedback_vectors.shape}")
    print(f"  Vector norms: {np.linalg.norm(feedback_vectors, axis=1)}")
    print()
    
    # Initialize model
    print("Initializing HyperColumn model...")
    model = HyperColumnGPU(
        input_len=INPUT_LEN,
        minicolumn_count=MINICOLUMN_COUNT,
        alpha=ALPHA,
        beta=BETA,
        feedback_ratio=FEEDBACK_RATIO,
        is_learning=True
    )
    print(f"  Weights shape: {(MINICOLUMN_COUNT, INPUT_LEN**2)}")
    print()
    
    # Training loop
    print("=" * 60)
    print("Training")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Shuffle training data
        indices = np.random.permutation(len(train_images))
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels[indices]
        
        # Train
        recon_error, cos_sim, class_acc = train_epoch(
            model=model,
            images=train_images_shuffled,
            labels=train_labels_shuffled,
            feedback_vectors=feedback_vectors,
            batch_size=BATCH_SIZE,
            epoch=epoch
        )
        
        print(f"  Cosine Sim: {cos_sim:.4f}, Recon Error: {recon_error:.6f}, Class Acc: {class_acc:.2%}")
    
    print()
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_recon_error, train_cos_sim, train_class_acc = evaluate(
        model=model,
        images=train_images,
        labels=train_labels,
        feedback_vectors=feedback_vectors,
        batch_size=BATCH_SIZE
    )
    print(f"  Cosine Sim: {train_cos_sim:.4f}, Recon Error: {train_recon_error:.6f}, Class Acc: {train_class_acc:.2%}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_recon_error, test_cos_sim, test_class_acc = evaluate(
        model=model,
        images=test_images,
        labels=test_labels,
        feedback_vectors=feedback_vectors,
        batch_size=BATCH_SIZE
    )
    print(f"  Cosine Sim: {test_cos_sim:.4f}, Recon Error: {test_recon_error:.6f}, Class Acc: {test_class_acc:.2%}")
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Training - Cosine Sim: {train_cos_sim:.4f}, Recon Error: {train_recon_error:.6f}, Class Acc: {train_class_acc:.2%}")
    print(f"  Test     - Cosine Sim: {test_cos_sim:.4f}, Recon Error: {test_recon_error:.6f}, Class Acc: {test_class_acc:.2%}")
    print(f"  GPU Mode: {'Enabled' if GPU_AVAILABLE else 'CPU Fallback'}")


if __name__ == "__main__":
    main()
