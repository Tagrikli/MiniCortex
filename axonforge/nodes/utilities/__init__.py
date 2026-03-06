"""Utilities package for MiniCortex nodes."""

import numpy as np
from pathlib import Path
import threading

from PySide6.QtCore import QStandardPaths


_DATASET_LOAD_LOCK = threading.Lock()
_DATASET_CACHE = {}
_APP_NAME = "AxonForge"


def _resolve_dataset_cache_dir() -> Path:
    """Resolve persistent dataset cache directory using Qt cache location."""
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericCacheLocation
    )
    if not base:
        base = str(Path.home() / ".cache")
    cache_dir = Path(base) / _APP_NAME / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _mnist_cache_paths(dataset_name: str):
    cache_dir = _resolve_dataset_cache_dir()
    images_path = cache_dir / f"{dataset_name}_train_images.npy"
    labels_path = cache_dir / f"{dataset_name}_train_labels.npy"
    return images_path, labels_path


def _build_mnist_cache(dataset_name: str, data_path: Path, images_path: Path, labels_path: Path) -> bool:
    """Build cached .npy files in-process."""
    try:
        from mnist import MNIST

        mn = MNIST(str(data_path))
        images, labels = mn.load_training()
        images_np = np.array(images, dtype=np.float32).reshape(-1, 28, 28) / 255.0
        labels_np = np.array(labels, dtype=np.int64)
        np.save(images_path, images_np)
        np.save(labels_path, labels_np)
        return True
    except Exception as e:
        print(f"Warning: failed to build {dataset_name} cache: {e}")
        return False


def _load_dataset_with_python_mnist(dataset_name: str = "mnist"):
    """
    Load MNIST dataset using python-mnist.
    
    Args:
        dataset_name: Either "mnist" or "fashion_mnist"
    
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    # Determine data directory - go up 4 levels from axonforge/nodes/utilities/__init__.py
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

    with _DATASET_LOAD_LOCK:
        cached = _DATASET_CACHE.get(dataset_name)
        if cached is not None:
            return cached

        images_path, labels_path = _mnist_cache_paths(dataset_name)
        if not images_path.exists() or not labels_path.exists():
            built = _build_mnist_cache(dataset_name, data_path, images_path, labels_path)
            if not built:
                return None, None

        try:
            # Use mmap for fast, low-overhead shared reads across nodes.
            images = np.load(images_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
        except Exception as e:
            print(f"Warning: failed to load cached {dataset_name} arrays: {e}")
            return None, None

        _DATASET_CACHE[dataset_name] = (images, labels)
        return images, labels
