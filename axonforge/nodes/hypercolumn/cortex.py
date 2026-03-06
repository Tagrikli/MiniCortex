import math
import numpy as np
from typing import Optional

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Range, Integer, Bool
from axonforge.core.descriptors.displays import Vector2D, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch

class Reconstruct(Node):
    """Reconstruct an input from weights and activations: x_reconstructed = W.T @ a.

    Takes the (n, D) weight matrix and the (k, k) activation map from a
    HyperColumn, flattens the activations to (n,), computes W.T @ a → (D,),
    and reshapes back to the original 2-D input shape for output and display.
    """

    # ── Ports ──────────────────────────────────────────────────────────
    weights = InputPort("Weights", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    output = OutputPort("Output", np.ndarray)

    # ── Displays ───────────────────────────────────────────────────────
    preview = Vector2D("Reconstruction", color_mode="grayscale")
    info = Text("Info", default="Waiting for inputs…")

    def process(self):
        if self.weights is None or self.activations is None:
            return

        W = self.weights  # (n, D)
        a = self.activations  # (k, k) or any shape with n total elements

        n, D = W.shape
        a_flat = a.flatten().astype(np.float64)  # (n,)

        x_recon = W.T @ a_flat

        side = int(np.sqrt(D))
        x_recon_2d = x_recon.reshape(side, side)

        self.output = x_recon_2d
        self.preview = x_recon_2d
        self.info = (
            f"W ({n}×{D}) · a ({n},) → ({x_recon_2d.shape[0]}×{x_recon_2d.shape[1]})"
        )
