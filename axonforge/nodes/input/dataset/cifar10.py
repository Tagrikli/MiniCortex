"""CIFAR-10 dataset input node for MiniCortex."""

import numpy as np
from pathlib import Path

from ....core.node import Node, background_init
from ....core.descriptors.ports import InputPort, OutputPort
from ....core.descriptors.properties import Integer
from ....core.descriptors.displays import Vector2D, Text
from ....core.descriptors.store import Store
from ....core.descriptors import branch



class InputCIFAR10(Node):
    """Cycle through CIFAR-10 images.
    
    CIFAR-10 dataset contains 32x32 color images in 10 classes:
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """
    # Full RGB output
    output_pattern = OutputPort("Pattern", np.ndarray)
    output_class = OutputPort("Class", int)
    pattern = Vector2D("Pattern", color_mode="rgb")
    class_label = Text("Class", default="airplane")
    
    # Opponent color space outputs (human visual system inspired)
    output_rg = OutputPort("RG (R-G)", np.ndarray)      # Red-Green opponency
    output_by = OutputPort("BY (0.5*(R+G)-B)", np.ndarray)  # Blue-Yellow opponency
    output_lum = OutputPort("Luminance", np.ndarray)    # Luminance
    
    # Unified opponent color output (3 channels: RG, BY, LUM)
    output_opponent = OutputPort("Opponent (RG,BY,LUM)", np.ndarray)
    opponent_display = Vector2D("Opponent", color_mode="rgb")
    
    # Property: how many times to repeat each image
    repeat_count = Integer("Repeat", default=1)
    
    # Store: index and step counter
    idx = Store(default=0)
    repeat_counter = Store(default=0)
    size = Store(default=32)

    LABEL_NAMES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    @background_init
    def init(self):
        self._images, self._labels = self._load_cifar10()
        self._update_pattern()

    def _load_cifar10(self):
        """Load CIFAR-10 dataset from data directory."""
        # Go up from axonforge/nodes/input/dataset/ to project root
        data_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "cifar10"
        
        # Try to load from numpy format first
        train_images_path = data_path / "train_images.npy"
        train_labels_path = data_path / "train_labels.npy"
        
        if train_images_path.exists() and train_labels_path.exists():
            images = np.load(train_images_path)
            labels = np.load(train_labels_path)
            return images, labels
        
        # Try to load from pickle format (original CIFAR-10 format)
        try:
            import pickle
            
            images = []
            labels = []
            
            # Load training batches (data_batch_1 to data_batch_5)
            for i in range(1, 6):
                batch_path = data_path / f"data_batch_{i}"
                if batch_path.exists():
                    with open(batch_path, 'rb') as f:
                        batch = pickle.load(f, encoding='bytes')
                        images.append(batch[b'data'])
                        labels.extend(batch[b'labels'])
            
            if images:
                # Reshape: N x 3072 -> N x 3 x 32 x 32 -> N x 32 x 32 x 3 (RGB)
                images = np.vstack(images)
                images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                images = images.astype(np.float32) / 255.0
                labels = np.array(labels)
                return images, labels
                
        except Exception as e:
            print(f"Error loading CIFAR-10: {e}")
        
        # Return empty if not found
        print(f"CIFAR-10 dataset not found at {data_path}")
        return None, None

    def process(self):
        if self._images is not None:
            # Increment repeat counter
            self.repeat_counter = int(self.repeat_counter) + 1
            
            # Only advance to next image when repeat_count is reached
            if int(self.repeat_counter) >= int(self.repeat_count):
                self.idx = (int(self.idx) + 1) % len(self._images)
                self.repeat_counter = 0
        
        self._update_pattern()

    def _update_pattern(self):
        if self._images is not None:
            image = self._images[self.idx]
            
            # Full RGB output
            self.output_pattern = image
            self.pattern = image
            self.output_class = int(self._labels[self.idx])
            class_idx = int(self._labels[self.idx])
            self.class_label = self.LABEL_NAMES[class_idx] if class_idx < len(self.LABEL_NAMES) else str(class_idx)
            
            # Extract RGB channels
            R = image[:, :, 0]
            G = image[:, :, 1]
            B = image[:, :, 2]
            
            # Compute opponent color space (human visual system inspired)
            # rg: Red-Green opponency (positive = red, negative = green)
            rg = R - G
            self.output_rg = rg
            
            # by: Blue-Yellow opponency (positive = blue, negative = yellow)
            # yellow is approximated as (R+G)/2
            by = 0.5 * (R + G) - B
            self.output_by = by
            
            # lum: Luminance (weighted sum, standard RGB to grayscale)
            lum = 0.3 * R + 0.59 * G + 0.11 * B
            self.output_lum = lum
            
            # Unified opponent color output (3 channels: RG, BY, LUM)
            # Stack into HxWx3 array for RGB-style processing
            self.output_opponent = np.stack([rg, by, lum], axis=2)
            self.opponent_display = self.output_opponent
            
