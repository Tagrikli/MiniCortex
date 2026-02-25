from typing import Dict, Optional, Any
from pathlib import Path
import os
import numpy as np

from ..core.node import Node
from ..core.descriptors.ports import InputPort, OutputPort
from ..core.descriptors.properties import Range, Integer, Bool, Enum
from ..core.descriptors.displays import Vector2D, Text
from ..core.descriptors.actions import Action
from ..core.descriptors.store import Store
from ..core.descriptors.node import node


def _load_dataset_with_python_mnist(dataset: str):
    from mnist import MNIST

    dataset_key = dataset.lower()
    env_var = (
        "MINICORTEX_MNIST_DIR"
        if dataset_key == "mnist"
        else "MINICORTEX_FASHION_MNIST_DIR"
    )
    candidates = []
    env_path = os.environ.get(env_var)
    if env_path:
        candidates.append(Path(env_path))
    if dataset_key == "mnist":
        candidates.extend(
            [
                Path("data/mnist/mnist"),
                Path("data/mnist"),
                Path("data/MNIST"),
                Path("datasets/mnist"),
                Path("mnist"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("data/mnist/fashion-mnist"),
                Path("data/mnist/fashion_mnist"),
                Path("data/fashion-mnist"),
                Path("data/fashion_mnist"),
                Path("data/FashionMNIST"),
                Path("datasets/fashion-mnist"),
                Path("fashion-mnist"),
            ]
        )

    last_error = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            mndata = MNIST(str(path))
            if any(path.glob("*.gz")):
                mndata.gz = True
            images, labels = mndata.load_training()
            images = np.asarray(images, dtype=np.float32).reshape((-1, 28, 28)) / 255.0
            return images, np.asarray(labels, dtype=np.int32)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load {dataset}. Set {env_var}.") from last_error





@node.input
class InputMovingShape(Node):
    """Generate a moving shape (square or circle) on a 2D grid."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    info = Text("Info", default="Position: (0, 0)")

    # Properties
    grid_size = Integer("Grid Size", default=64, min_val=8, max_val=256)
    shape_type = Enum("Shape", ["Square", "Circle"], default="Square")
    shape_size = Integer("Shape Size", default=8, min_val=1, max_val=64)
    speed = Range("Speed", default=0.1, min_val=0.01, max_val=1.0, scale="log")
    interpolation = Enum("Interpolation", ["Linear", "Ease In", "Ease Out", "Ease In-Out"], default="Linear")
    
    # Checkboxes for modes
    auto_move = Bool("Auto Move", default=False)

    # Buttons
    prev_pos = Action("Prev", callback="_on_prev")
    next_pos = Action("Next", callback="_on_next")

    # Store
    pos_x = Store(default=0.0)
    pos_y = Store(default=0.0)
    target_x = Store(default=0.0)
    target_y = Store(default=0.0)

    def init(self):
        self._generate_new_target()
        self._update_pattern()

    def process(self):
        if self.auto_move:
            self._move_towards_target()
        self._update_pattern()

    def _move_towards_target(self):
        """Move position towards target using interpolation."""
        speed = float(self.speed)
        current_x = float(self.pos_x)
        current_y = float(self.pos_y)
        target_x = float(self.target_x)
        target_y = float(self.target_y)
        
        # Calculate distance to target
        dx = target_x - current_x
        dy = target_y - current_y
        dist = np.sqrt(dx * dx + dy * dy)
        
        # Check if we've reached the target
        if dist < speed:
            self.pos_x = target_x
            self.pos_y = target_y
            # Generate new random target
            self._generate_new_target()
        else:
            # Move towards target
            self.pos_x = current_x + (dx / dist) * speed
            self.pos_y = current_y + (dy / dist) * speed

    def _generate_new_target(self):
        """Generate a new random target position."""
        grid_size = int(self.grid_size)
        shape_size = int(self.shape_size)
        margin = shape_size // 2 + 1
        self.target_x = np.random.uniform(margin, grid_size - margin - 1)
        self.target_y = np.random.uniform(margin, grid_size - margin - 1)

    def _apply_interpolation(self, t):
        """Apply interpolation function based on selected mode."""
        interp_mode = self.interpolation
        if interp_mode == "Linear":
            return t
        elif interp_mode == "Ease In":
            return t * t
        elif interp_mode == "Ease Out":
            return 1 - (1 - t) * (1 - t)
        elif interp_mode == "Ease In-Out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        return t

    def _update_pattern(self):
        grid_size = int(self.grid_size)
        shape_size = int(self.shape_size)
        pos_x = float(self.pos_x)
        pos_y = float(self.pos_y)

        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:grid_size, :grid_size]
        
        if self.shape_type == "Square":
            # Calculate distance from edges for anti-aliasing
            half_size = shape_size / 2.0
            dx = np.abs(x_coords - pos_x)
            dy = np.abs(y_coords - pos_y)
            
            # Anti-aliased square
            dist_x = half_size - dx
            dist_y = half_size - dy
            pattern = np.minimum(np.clip(dist_x, 0.0, 1.0), np.clip(dist_y, 0.0, 1.0))
        else:  # Circle
            # Calculate distance from center
            dx = x_coords - pos_x
            dy = y_coords - pos_y
            dist = np.sqrt(dx * dx + dy * dy)
            
            # Anti-aliased circle
            radius = shape_size / 2.0
            pattern = np.clip(radius - dist, 0.0, 1.0)

        self.output_pattern = pattern.astype(np.float32)
        self.pattern = pattern.astype(np.float32)
        
        # Update info
        px = int(pos_x)
        py = int(pos_y)
        if self.auto_move:
            tx = int(float(self.target_x))
            ty = int(float(self.target_y))
            self.info = f"Pos: ({px}, {py}) → ({tx}, {ty})"
        else:
            self.info = f"Pos: ({px}, {py})"

    def _on_prev(self, params):
        """Move to previous position (step back)."""
        step = 5
        self.pos_x = max(0, float(self.pos_x) - step)
        self._update_pattern()
        return {"status": "ok"}

    def _on_next(self, params):
        """Move to next position (step forward)."""
        step = 5
        grid_size = int(self.grid_size)
        self.pos_x = min(grid_size - 1, float(self.pos_x) + step)
        self._update_pattern()
        return {"status": "ok"}


@node.input
class InputDigitMNIST(Node):
    """Cycle through MNIST digit images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    output_digit = OutputPort("Digit", int)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    digit = Text("Digit", default="0")
    
    # Store only the index, images loaded from dataset (transient)
    idx = Store(default=0)
    size = Store(default=28)

    def init(self):
        self._images, self._labels = _load_dataset_with_python_mnist("mnist")
        self._update_pattern()

    def process(self):
        if self._images is not None:
            self.idx = (self.idx + 1) % len(self._images)
        self._update_pattern()

    def _update_pattern(self):
        if self._images is not None:
            self.output_pattern = self._images[self.idx]
            self.pattern = self._images[self.idx]
            self.output_digit = int(self._labels[self.idx])
            self.digit = str(self._labels[self.idx])


@node.input
class InputFashionMNIST(Node):
    """Cycle through Fashion-MNIST images."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    info = Text("Info", default="Fashion: 0")
    LABEL_NAMES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    
    # Store only the index, images loaded from dataset (transient)
    idx = Store(default=0)
    size = Store(default=28)

    def init(self):
        self._images, self._labels = _load_dataset_with_python_mnist("fashion_mnist")
        self._update_pattern()

    def process(self):
        if self._images is not None:
            self.idx = (self.idx + 1) % len(self._images)
        self._update_pattern()

    def _update_pattern(self):
        if self._images is not None:
            self.output_pattern = self._images[self.idx]
            self.pattern = self._images[self.idx]
            l = int(self._labels[self.idx])
            self.info = (
                f"Fashion: {self.LABEL_NAMES[l] if l < len(self.LABEL_NAMES) else l}"
            )


@node.input
class InputInteger(Node):
    """Output an integer value."""

    output_value = OutputPort("Value", int)
    value = Integer("Value", default=0)

    def init(self):
        self.output_value = 0

    def process(self):
        self.output_value = int(self.value)


@node.input
class InputRotatingLine(Node):
    """Generate a rotating line pattern on a square grid."""

    output_pattern = OutputPort("Pattern", np.ndarray)
    pattern = Vector2D("Pattern", color_mode="grayscale")
    info = Text("Info", default="Angle: 0°")

    # Properties
    size = Integer("Size", default=64, min_val=8, max_val=256)
    thickness = Integer("Thickness", default=2, min_val=1, max_val=20)
    rotation_speed = Range("Rotation Speed", default=0.1, min_val=0.001, max_val=1.0, scale="log")
    interpolation = Enum("Interpolation", ["Linear", "Ease In", "Ease Out", "Ease In-Out"], default="Linear")
    
    # Checkboxes for modes
    auto_rotate = Bool("Auto Rotate", default=False)
    random_mode = Bool("Random Mode", default=False)

    # Buttons
    prev_pattern = Action("Prev", callback="_on_prev")
    next_pattern = Action("Next", callback="_on_next")

    # Store
    angle = Store(default=0.0)
    target_angle = Store(default=0.0)
    interp_progress = Store(default=0.0)
    start_angle = Store(default=0.0)

    def init(self):
        self._generate_new_target()
        self._update_pattern()

    def process(self):
        if self.random_mode:
            # Move towards target angle using shortest path
            self._move_towards_target()
        elif self.auto_rotate:
            # Continuous rotation
            speed = float(self.rotation_speed)
            self.angle = (self.angle + speed) % (2 * np.pi)
        
        self._update_pattern()

    def _move_towards_target(self):
        """Move angle towards target using shortest path with interpolation."""
        speed = float(self.rotation_speed)
        current = float(self.angle)
        target = float(self.target_angle)
        
        # Calculate shortest path direction
        diff = target - current
        # Normalize to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        # Check if we've reached the target
        if abs(diff) < speed:
            self.angle = target
            # Generate new random target
            self._generate_new_target()
        else:
            # Move in the shortest direction
            direction = 1 if diff > 0 else -1
            self.angle = (current + direction * speed) % (2 * np.pi)

    def _generate_new_target(self):
        """Generate a new random target angle."""
        self.target_angle = np.random.uniform(0, 2 * np.pi)

    def _apply_interpolation(self, t):
        """Apply interpolation function based on selected mode."""
        interp_mode = self.interpolation
        if interp_mode == "Linear":
            return t
        elif interp_mode == "Ease In":
            return t * t
        elif interp_mode == "Ease Out":
            return 1 - (1 - t) * (1 - t)
        elif interp_mode == "Ease In-Out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        return t

    def _update_pattern(self):
        size = int(self.size)
        thickness = int(self.thickness)
        angle = float(self.angle)

        # Create coordinate grids centered at the middle
        y_coords, x_coords = np.ogrid[:size, :size]
        cx, cy = size / 2.0, size / 2.0
        rx = x_coords - cx
        ry = y_coords - cy

        # Calculate perpendicular distance to the line through center at given angle
        # Line direction: (cos(angle), sin(angle))
        # Perpendicular distance: |rx * sin(angle) - ry * cos(angle)|
        line_dx = np.cos(angle)
        line_dy = np.sin(angle)
        perp_dist = np.abs(rx * line_dy - ry * line_dx)

        # Create line pattern with anti-aliasing at edges
        half_thickness = thickness / 2.0
        pattern = np.clip(half_thickness - perp_dist, 0.0, 1.0)

        self.output_pattern = pattern.astype(np.float32)
        self.pattern = pattern.astype(np.float32)
        
        angle_deg = int(np.degrees(angle)) % 360
        if self.random_mode:
            target_deg = int(np.degrees(float(self.target_angle))) % 360
            self.info = f"Angle: {angle_deg}° → {target_deg}°"
        elif self.auto_rotate:
            self.info = f"Angle: {angle_deg}° | Auto"
        else:
            self.info = f"Angle: {angle_deg}°"

    def _on_prev(self, params):
        """Step angle backwards by 22.5 degrees."""
        step = np.pi / 8  # 22.5 degrees
        self.angle = (self.angle - step) % (2 * np.pi)
        self._update_pattern()
        return {"status": "ok"}

    def _on_next(self, params):
        """Step angle forwards by 22.5 degrees."""
        step = np.pi / 8  # 22.5 degrees
        self.angle = (self.angle + step) % (2 * np.pi)
        self._update_pattern()
        return {"status": "ok"}

