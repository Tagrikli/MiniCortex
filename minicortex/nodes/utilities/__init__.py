"""Utilities package for MiniCortex nodes."""

import numpy as np
from pathlib import Path


def _load_dataset_with_python_mnist(dataset_name: str = "mnist"):
    """
    Load MNIST dataset using python-mnist.
    
    Args:
        dataset_name: Either "mnist" or "fashion_mnist"
    
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    try:
        from mnist import MNIST
    except ImportError:
        print("Warning: python-mnist not installed. Install with: pip install python-mnist")
        return None, None
    
    # Determine data directory - go up 4 levels from minicortex/nodes/utilities/__init__.py
    # to reach project root, then go to data/
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    if dataset_name == "mnist":
        # Download script puts MNIST in data/mnist/mnist/
        data_path = data_dir / "mnist" / "mnist"
    elif dataset_name == "fashion_mnist":
        # Download script puts Fashion-MNIST in data/mnist/fashion-mnist/
        data_path = data_dir / "mnist" / "fashion-mnist"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not data_path.exists():
        print(f"Warning: Dataset not found at {data_path}. Run utils/download_mnist_datasets.py first.")
        print(f"  Expected path: {data_path.absolute()}")
        # Try legacy paths for backward compatibility
        if dataset_name == "mnist":
            legacy_path = data_dir / "mnist_data"
        else:
            legacy_path = data_dir / "fashion_mnist_data"
        
        if legacy_path.exists():
            print(f"  Found dataset at legacy path: {legacy_path}")
            data_path = legacy_path
        else:
            return None, None
    
    # Load dataset
    mnist_data = MNIST(str(data_path))
    
    if dataset_name == "mnist":
        images, labels = mnist_data.load_training()
    else:
        images, labels = mnist_data.load_training()
    
    # Convert to numpy arrays
    images = np.array(images).reshape(-1, 28, 28).astype(np.float32) / 255.0
    labels = np.array(labels)
    
    return images, labels
