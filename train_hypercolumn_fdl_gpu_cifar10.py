#!/usr/bin/env python3
"""
GPU-accelerated HyperColumn FDL (Feedback-Driven Learning) training script using CuPY.

Replicates the HyperColumnFeedbackDriven.process() function for GPU calculations.
Trains on CIFAR-10 dataset with class-specific feedback vectors and RG/BY/LUM color space.

Color space conversion:
- RG = R - G (Red-Green opponency)
- BY = 0.5*(R+G) - B (Blue-Yellow opponency)
- LUM = 0.3*R + 0.59*G + 0.11*B (Luminance)

Metrics reported:
- Cosine similarity between output activations and injected feedback
- Reconstruction error (MSE)
- Classification accuracy
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

# Try to import CuPY for GPU acceleration, fallback to NumPy if unavailable
try:
    import cupy as cp
    from cupy.cuda import Device
    GPU_AVAILABLE = True
    print("CuPY imported successfully - GPU acceleration enabled")
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
# GLOBAL PARAMETERS (from HyperColumnFeedbackDriven class defaults)
# ==============================================================================
INPUT_SIZE = 32                   # Input image size (32x32 for CIFAR-10)
NUM_CHANNELS = 3                  # RG, BY, LUM channels
INPUT_LEN = INPUT_SIZE            # For compatibility with existing code
MINICOLUMN_COUNT = 16             # Number of minicolumns
ALPHA = 0.003                     # Learning rate (α)
BETA = 0.001                      # Inhibition parameter (β)
EPS = 1e-8                        # Small constant for numerical stability
NUM_CLASSES = 10                  # CIFAR-10 has 10 classes
BATCH_SIZE = 32                   # Training batch size
NUM_EPOCHS = 1                    # Number of training epochs
NOISE_STD = 0.0                   # Gaussian noise standard deviation (0.0 = no noise)

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_cifar10_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 dataset from pickle files."""
    data_path = Path(__file__).parent / "data" / "cifar10"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"CIFAR-10 dataset not found at {data_path}. "
            "Run utils/download_cifar10.py first."
        )
    
    # Load training batches (data_batch_1 to data_batch_5)
    train_images = []
    train_labels = []
    
    for i in range(1, 6):
        batch_path = data_path / f"data_batch_{i}"
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            train_images.append(batch[b'data'])
            train_labels.extend(batch[b'labels'])
    
    # Load test batch
    test_path = data_path / "test_batch"
    with open(test_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
        test_images = test_batch[b'data']
        test_labels = test_batch[b'labels']
    
    # Concatenate training data
    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    print(f"Loaded CIFAR-10 dataset:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    return train_images, train_labels, test_images, test_labels


def convert_rgb_to_rg_by_lum(rgb_flat: np.ndarray) -> np.ndarray:
    """
    Convert RGB images to RG/BY/LUM color space and flatten.
    
    Args:
        rgb_flat: Flattened RGB images, shape (N, 3072) where 3072 = 32*32*3
                  Layout is [R channel (1024), G channel (1024), B channel (1024)]
    
    Returns:
        Flattened RG/BY/LUM images, shape (N, 3072)
        Layout is [RG channel (1024), BY channel (1024), LUM channel (1024)]
    """
    n_samples = rgb_flat.shape[0]
    pixels_per_channel = 32 * 32  # 1024
    
    # Extract R, G, B channels (CIFAR-10 format: R, G, B each 1024 values)
    R = rgb_flat[:, :pixels_per_channel].reshape(n_samples, 32, 32)
    G = rgb_flat[:, pixels_per_channel:2*pixels_per_channel].reshape(n_samples, 32, 32)
    B = rgb_flat[:, 2*pixels_per_channel:].reshape(n_samples, 32, 32)
    
    # Normalize to [0, 1]
    R = R.astype(np.float32) / 255.0
    G = G.astype(np.float32) / 255.0
    B = B.astype(np.float32) / 255.0
    
    # Convert to RG/BY/LUM color space
    # rg = R - G (Red-Green opponency)
    rg = R - G
    
    # by = 0.5*(R+G) - B (Blue-Yellow opponency)
    by = 0.5 * (R + G) - B
    
    # lum = 0.3*R + 0.59*G + 0.11*B (Luminance)
    lum = 0.3 * R + 0.59 * G + 0.11 * B
    
    # Flatten and concatenate: RG, BY, LUM
    rg_flat = rg.reshape(n_samples, -1)
    by_flat = by.reshape(n_samples, -1)
    lum_flat = lum.reshape(n_samples, -1)
    
    opponent_flat = np.concatenate([rg_flat, by_flat, lum_flat], axis=1)
    
    return opponent_flat


def create_class_feedback_vectors(vector_size: int, num_classes: int, seed: int = 42) -> np.ndarray:
    """Create random unit vectors for each class to use as feedback."""
    rng = np.random.RandomState(seed)
    feedback_vectors = rng.randn(num_classes, vector_size).astype(np.float32)
    norms = np.linalg.norm(feedback_vectors, axis=1, keepdims=True) + EPS
    feedback_vectors = feedback_vectors / norms
    return feedback_vectors


def add_gaussian_noise(images: np.ndarray, noise_std: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Add Gaussian noise to images.
    
    Args:
        images: Input images, shape (batch, features)
        noise_std: Standard deviation of Gaussian noise
        seed: Optional random seed for reproducibility
        
    Returns:
        Noisy images with same shape as input
    """
    if noise_std <= 0:
        return images
    
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, size=images.shape).astype(np.float32)
    noisy_images = images + noise
    return noisy_images


# ==============================================================================
# GPU-ACCELERATED HYPERCOLUMN FDL PROCESS FUNCTION
# ==============================================================================

class HyperColumnFDLGPU:
    """
    GPU-accelerated HyperColumn FDL (Feedback-Driven Learning) implementation.
    Replicates the HyperColumnFeedbackDriven.process() function.
    """
    
    def __init__(
        self,
        input_len: int = INPUT_LEN,
        num_channels: int = NUM_CHANNELS,
        minicolumn_count: int = MINICOLUMN_COUNT,
        alpha: float = ALPHA,
        beta: float = BETA,
        is_learning: bool = True
    ):
        self.input_len = input_len
        self.num_channels = num_channels
        self.input_features = input_len * input_len * num_channels  # 32*32*3 = 3072
        self.minicolumn_count = minicolumn_count
        self.alpha = alpha
        self.beta = beta
        self.is_learning = is_learning
        
        # Initialize weights on GPU
        self.weights = self._initialize_weights()
    
    def _initialize_weights(self) -> cp.ndarray:
        """Initialize random unit vectors as weights on GPU."""
        weights = cp.random.randn(self.minicolumn_count, self.input_features).astype(cp.float32)
        weights /= cp.linalg.norm(weights, axis=1, keepdims=True) + EPS
        return weights
    
    def calculate_delta_rot_matrix(self) -> cp.ndarray:
        """Calculate angular proximity matrix between weight vectors."""
        dots = self.weights @ self.weights.T
        delta_rot = cp.clip(dots, 0.0, 1.0)
        cp.fill_diagonal(delta_rot, 0.0)
        return delta_rot
    
    def process(
        self,
        x_input: cp.ndarray,
        feedback: Optional[cp.ndarray] = None,
        return_reconstruction: bool = False
    ) -> cp.ndarray | Tuple[cp.ndarray, cp.ndarray]:
        """
        Process a single input through the HyperColumn FDL.
        
        Replicates the HyperColumnFeedbackDriven.process() function.
        
        Args:
            x_input: Input array, shape (input_features,) = (3072,) for flattened RG/BY/LUM
            feedback: Optional feedback vector, shape (minicolumn_count,)
            return_reconstruction: If True, also return reconstructed input
            
        Returns:
            Output activations s_final, shape (minicolumn_count,)
            If return_reconstruction is True, returns (output, x_reconstructed)
        """
        eps = EPS
        
        ######## INPUT ########
        if x_input.ndim > 1:
            x = x_input.flatten()
        else:
            x = x_input
        
        x_norm_factor = cp.linalg.norm(x) + eps
        x_norm = x / x_norm_factor
        
        # Keep unit templates
        self.weights = self.weights / (cp.linalg.norm(self.weights, axis=1, keepdims=True) + eps)
        
        N = self.weights.shape[0]
        
        ######## FEEDBACK (signed) -> normalized match + separate scale ########
        if feedback is None:
            # No feedback: neutral "all on"
            f = cp.ones(N, dtype=self.weights.dtype)
            f_scale = 1.0
            f_hat = f
        else:
            f = feedback.flatten().astype(self.weights.dtype)
            # Global feedback magnitude
            f_scale = float(cp.linalg.norm(f) + eps)
            f_hat = f / f_scale  # Signed, L2-normalized feedback direction
        
        # Match comes from normalized feedback
        match = f_hat
        match_pos = cp.maximum(match, 0.0)  # Attention/learning only for positive match
        
        ######## DELTA ROT (angular proximity) ########
        delta_rot = self.calculate_delta_rot_matrix()  # (N,N) in [0,1], diag=0
        
        ######## DENSITY (per-template crowding) ########
        density = delta_rot.sum(axis=1)  # (N,)
        density = density / (density.mean() + eps)  # Stabilize across N
        
        ######## FEEDFORWARD ########
        s_raw = x_norm @ self.weights.T  # (N,)
        
        # Apply feedback match to expression
        s_fb = s_raw * (match_pos * f_scale)
        
        # Lateral suppression ONLY when close (delta_rot) and ONLY from positive expressed activity
        inh = self.beta * (delta_rot @ cp.maximum(s_fb, 0.0))
        s_final = s_fb - inh
        
        ######## LEARNING (feedback-driven; slowed by density; repels when crowded) ########
        if self.is_learning and (feedback is not None):
            # Attraction rotation magnitude
            theta_attr = self.alpha * (match_pos * f_scale) / (1.0 + self.beta * density)  # (N,)
            
            # Tangent direction toward x_norm on the unit sphere
            dots_x = (self.weights @ x_norm)[:, None]  # (N,1)
            tangent = x_norm[None, :] - dots_x * self.weights
            tangent_hat = tangent / (cp.linalg.norm(tangent, axis=1, keepdims=True) + eps)
            
            # Repulsion direction away from nearby templates (pure geometry)
            rep = delta_rot @ self.weights
            rep = rep - (cp.sum(rep * self.weights, axis=1, keepdims=True) * self.weights)  # Tangent proj
            rep_hat = rep / (cp.linalg.norm(rep, axis=1, keepdims=True) + eps)
            
            # Repulsion magnitude
            theta_rep = self.alpha * (match_pos * f_scale) * (self.beta * density)  # (N,)
            
            # Combine into one on-sphere rotation step
            t = (theta_attr[:, None] * tangent_hat) - (theta_rep[:, None] * rep_hat)
            t_norm = cp.linalg.norm(t, axis=1, keepdims=True) + eps
            t_hat = t / t_norm
            theta = t_norm[:, 0]
            
            self.weights = cp.cos(theta)[:, None] * self.weights + cp.sin(theta)[:, None] * t_hat
            self.weights = self.weights / (cp.linalg.norm(self.weights, axis=1, keepdims=True) + eps)
            
            # Recompute activations after learning
            s_raw = x_norm @ self.weights.T
            s_fb = s_raw * (match_pos * f_scale)
            inh = self.beta * (delta_rot @ cp.maximum(s_fb, 0.0))
            s_final = s_fb - inh
        
        ######## OUTPUT ########
        output = s_final * x_norm_factor
        
        if return_reconstruction:
            # Reconstruct input from activations
            x_recon = self.weights.T @ s_final
            return output, x_recon
        
        return output
    
    def process_batch(
        self,
        x_batch: cp.ndarray,
        feedback_batch: Optional[cp.ndarray] = None,
        return_reconstruction_errors: bool = False
    ) -> cp.ndarray | Tuple[cp.ndarray, cp.ndarray]:
        """Process a batch of inputs sequentially."""
        batch_size = x_batch.shape[0]
        outputs = cp.zeros((batch_size, self.minicolumn_count), dtype=cp.float32)
        
        if return_reconstruction_errors:
            recon_errors = cp.zeros(batch_size, dtype=cp.float32)
            for i in range(batch_size):
                # Get feedback for this sample if available
                feedback = feedback_batch[i] if feedback_batch is not None else None
                output, x_recon = self.process(
                    x_batch[i], 
                    feedback, 
                    return_reconstruction=True
                )
                outputs[i] = output
                # Calculate reconstruction error (MSE)
                x_norm = x_batch[i] / (cp.linalg.norm(x_batch[i]) + EPS)
                recon_errors[i] = cp.mean((x_norm - x_recon) ** 2)
            return outputs, recon_errors
        else:
            for i in range(batch_size):
                feedback = feedback_batch[i] if feedback_batch is not None else None
                outputs[i] = self.process(x_batch[i], feedback)
            return outputs


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def calculate_cosine_similarity(
    outputs: cp.ndarray,
    feedbacks: cp.ndarray
) -> float:
    """Calculate mean cosine similarity between outputs and feedbacks."""
    output_norms = cp.linalg.norm(outputs, axis=1, keepdims=True) + EPS
    feedback_norms = cp.linalg.norm(feedbacks, axis=1, keepdims=True) + EPS
    outputs_norm = outputs / output_norms
    feedbacks_norm = feedbacks / feedback_norms
    similarities = cp.sum(outputs_norm * feedbacks_norm, axis=1)
    return float(cp.mean(similarities))


def calculate_classification_accuracy(
    outputs: cp.ndarray,
    labels: np.ndarray,
    feedback_vectors: np.ndarray
) -> float:
    """Calculate classification accuracy by comparing outputs to all class feedback vectors."""
    batch_size = outputs.shape[0]
    
    output_norms = cp.linalg.norm(outputs, axis=1, keepdims=True) + EPS
    outputs_norm = outputs / output_norms
    
    feedback_vectors_gpu = cp.asarray(feedback_vectors)
    feedback_norms = cp.linalg.norm(feedback_vectors_gpu, axis=1, keepdims=True) + EPS
    feedbacks_norm = feedback_vectors_gpu / feedback_norms
    
    similarities = outputs_norm @ feedbacks_norm.T
    predictions = cp.argmax(similarities, axis=1)
    
    labels_gpu = cp.asarray(labels)
    correct = cp.sum(predictions == labels_gpu)
    
    return float(correct / batch_size)


def train_epoch(
    model: HyperColumnFDLGPU,
    images: np.ndarray,
    labels: np.ndarray,
    feedback_vectors: np.ndarray,
    batch_size: int,
    epoch: int,
    noise_std: float = 0.0
) -> Tuple[float, float]:
    """Train for one epoch."""
    num_samples = len(images)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_similarity = 0.0
    total_recon_error = 0.0
    
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
    
    for batch_idx in pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Convert RGB to RG/BY/LUM color space
        batch_images_opponent = convert_rgb_to_rg_by_lum(batch_images)
        
        # Add Gaussian noise if specified
        if noise_std > 0:
            batch_images_opponent = add_gaussian_noise(batch_images_opponent, noise_std, seed=None)
        
        batch_feedbacks = feedback_vectors[batch_labels]
        
        batch_images_gpu = cp.asarray(batch_images_opponent)
        batch_feedbacks_gpu = cp.asarray(batch_feedbacks)
        
        outputs, recon_errors = model.process_batch(
            batch_images_gpu,
            batch_feedbacks_gpu,
            return_reconstruction_errors=True
        )
        
        similarity = calculate_cosine_similarity(outputs, batch_feedbacks_gpu)
        total_similarity += similarity * current_batch_size
        
        batch_recon_error = float(cp.mean(recon_errors))
        total_recon_error += batch_recon_error * current_batch_size
        
        pbar.set_postfix({
            'cos_sim': f'{similarity:.4f}',
            'recon_err': f'{batch_recon_error:.6f}'
        })
    
    mean_similarity = total_similarity / num_samples
    mean_recon_error = total_recon_error / num_samples
    
    return mean_recon_error, mean_similarity


def evaluate(
    model: HyperColumnFDLGPU,
    images: np.ndarray,
    labels: np.ndarray,
    feedback_vectors: np.ndarray,
    batch_size: int,
    noise_std: float = 0.0
) -> Tuple[float, float, float]:
    """
    Evaluate the model on a dataset WITHOUT feedback injection.
    
    This is pure feedforward inference - the model processes inputs
    without knowing the ground truth class labels.
    """
    num_samples = len(images)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_similarity = 0.0
    total_recon_error = 0.0
    total_correct = 0
    
    original_learning = model.is_learning
    model.is_learning = False
    
    pbar = tqdm(range(num_batches), desc="Evaluating (no feedback)")
    
    for batch_idx in pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Convert RGB to RG/BY/LUM color space
        batch_images_opponent = convert_rgb_to_rg_by_lum(batch_images)
        
        # Add Gaussian noise if specified
        if noise_std > 0:
            batch_images_opponent = add_gaussian_noise(batch_images_opponent, noise_std, seed=None)
        
        batch_feedbacks = feedback_vectors[batch_labels]
        
        batch_images_gpu = cp.asarray(batch_images_opponent)
        batch_feedbacks_gpu = cp.asarray(batch_feedbacks)
        
        # NO FEEDBACK INJECTION - pass None for feedback
        outputs, recon_errors = model.process_batch(
            batch_images_gpu, 
            feedback_batch=None,  # No feedback during evaluation
            return_reconstruction_errors=True
        )
        
        # Calculate cosine similarity between output and ground truth feedback (for reference)
        similarity = calculate_cosine_similarity(outputs, batch_feedbacks_gpu)
        total_similarity += similarity * current_batch_size
        
        batch_recon_error = float(cp.mean(recon_errors))
        total_recon_error += batch_recon_error * current_batch_size
        
        # Classification accuracy (compare outputs to all class feedbacks)
        batch_accuracy = calculate_classification_accuracy(
            outputs, batch_labels, feedback_vectors
        )
        total_correct += batch_accuracy * current_batch_size
        
        pbar.set_postfix({
            'cos_sim': f'{similarity:.4f}',
            'recon_err': f'{batch_recon_error:.6f}',
            'acc': f'{batch_accuracy:.2%}'
        })
    
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
    print("HyperColumn FDL GPU Training - CIFAR-10 (RG/BY/LUM)")
    print("=" * 60)
    print()
    
    print("Configuration:")
    print(f"  GPU Mode: {'Enabled' if GPU_AVAILABLE else 'CPU Fallback'}")
    print(f"  Input Size: {INPUT_SIZE}x{INPUT_SIZE}x{NUM_CHANNELS} = {INPUT_SIZE*INPUT_SIZE*NUM_CHANNELS}")
    print(f"  Minicolumns: {MINICOLUMN_COUNT}")
    print(f"  Alpha (learning rate): {ALPHA}")
    print(f"  Beta (inhibition): {BETA}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Noise Std: {NOISE_STD}")
    print()
    
    print("Loading CIFAR-10 dataset...")
    train_images, train_labels, test_images, test_labels = load_cifar10_dataset()
    print()
    
    print("Creating class feedback vectors...")
    feedback_vectors = create_class_feedback_vectors(
        vector_size=MINICOLUMN_COUNT,
        num_classes=NUM_CLASSES,
        seed=42
    )
    print(f"  Feedback vectors shape: {feedback_vectors.shape}")
    print(f"  Vector norms: {np.linalg.norm(feedback_vectors, axis=1)}")
    print()
    
    print("Initializing HyperColumn FDL model...")
    model = HyperColumnFDLGPU(
        input_len=INPUT_LEN,
        num_channels=NUM_CHANNELS,
        minicolumn_count=MINICOLUMN_COUNT,
        alpha=ALPHA,
        beta=BETA,
        is_learning=True
    )
    print(f"  Weights shape: {(MINICOLUMN_COUNT, INPUT_SIZE*INPUT_SIZE*NUM_CHANNELS)}")
    print()
    
    print("=" * 60)
    print("Training")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        indices = np.random.permutation(len(train_images))
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels[indices]
        
        recon_error, cos_sim = train_epoch(
            model=model,
            images=train_images_shuffled,
            labels=train_labels_shuffled,
            feedback_vectors=feedback_vectors,
            batch_size=BATCH_SIZE,
            epoch=epoch,
            noise_std=NOISE_STD
        )
        
        print(f"  Cosine Sim: {cos_sim:.4f}, Recon Error: {recon_error:.6f}")
    
    print()
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    print("\nEvaluating on training set (NO FEEDBACK - pure inference)...")
    train_recon_error, train_cos_sim, train_class_acc = evaluate(
        model=model,
        images=train_images,
        labels=train_labels,
        feedback_vectors=feedback_vectors,
        batch_size=BATCH_SIZE,
        noise_std=NOISE_STD
    )
    print(f"  Cosine Sim: {train_cos_sim:.4f}, Recon Error: {train_recon_error:.6f}, Class Acc: {train_class_acc:.2%}")
    
    print("\nEvaluating on test set (NO FEEDBACK - pure inference)...")
    test_recon_error, test_cos_sim, test_class_acc = evaluate(
        model=model,
        images=test_images,
        labels=test_labels,
        feedback_vectors=feedback_vectors,
        batch_size=BATCH_SIZE,
        noise_std=NOISE_STD
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
