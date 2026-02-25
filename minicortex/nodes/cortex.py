"""Cortical computation nodes for MiniCortex."""

import numpy as np
from typing import Optional, Dict, Any

from ..core.node import Node
from ..core.descriptors.ports import InputPort, OutputPort
from ..core.descriptors.properties import Range, Integer
from ..core.descriptors.displays import Vector2D, Text, Numeric
from ..core.descriptors.actions import Action
from ..core.descriptors.store import Store
from ..core.descriptors import dynamic, node


def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors.

    Both v0 and v1 must be unit-length (L2-normalized) 1-D arrays.
    Returns a unit-length vector that lies on the great-circle arc
    between v0 and v1, parameterised by *t* ∈ [0, 1].
    """
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = np.arccos(dot)

    # When the angle is tiny the sine denominator goes to zero;
    # fall back to a simple lerp (which is practically identical).
    if omega < 1e-10:
        result = (1.0 - t) * v0 + t * v1
    else:
        sin_omega = np.sin(omega)
        result = (
            (np.sin((1.0 - t) * omega) / sin_omega) * v0
            + (np.sin(t * omega) / sin_omega) * v1
        )

    # Re-normalise to stay exactly on the unit sphere.
    norm = np.linalg.norm(result)
    if norm > 1e-10:
        result /= norm
    return result


@dynamic
@node.custom("Cortex")
class HyperColumn(Node):
    """A hypercolumn containing *n* minicolumns with attraction/repulsion dynamics.

    Each minicolumn holds a unit weight vector the same size as the flattened
    input.  On every tick, each weight is:

    * **attracted** toward the L2-normalised input (controlled by *alpha*), and
    * **repelled** from the other minicolumns (controlled by *beta*).

    Both forces are computed in the tangent plane at the weight and the result
    is projected back onto the unit sphere via normalisation.

    The display output arranges every minicolumn's weight (reshaped to the
    original 2-D input shape) in a square grid.
    """

    # ── Ports ──────────────────────────────────────────────────────────
    input = InputPort("Input", np.ndarray)
    output = OutputPort("Output", np.ndarray)
    weights_out = OutputPort("Weights", np.ndarray)

    # ── Properties ─────────────────────────────────────────────────────
    minicolumn_count = Integer("Minicolumns", default=9, min_val=1)
    mean = Range("Mean", 0.0, -5.0, 5.0, scale="linear")
    std = Range("Std", 1.0, 0.01, 5.0, scale="log")
    alpha = Range("Alpha", 0.01, 0.001, 1.0, scale="log")
    beta = Range("Beta", 0.005, 0.001, 1.0, scale="log")

    # ── Displays ───────────────────────────────────────────────────────
    weights_display = Vector2D("Weights", color_mode="bwr")
    info = Text("Info", default="Waiting for input…")

    # ── Actions ────────────────────────────────────────────────────────
    reset_weights = Action("Reset Weights", callback="_on_reset_weights")

    # ── Stores ─────────────────────────────────────────────────────────
    _stored_input_shape = Store(default=None)

    # ──────────────────────────────────────────────────────────────────

    def init(self):
        self._weights: Optional[np.ndarray] = None  # (n, H*W) unit vectors
        self._input_shape: Optional[tuple] = None

        # Restore from a previous session / hot-reload if shape is known.
        if self._stored_input_shape is not None:
            self._input_shape = tuple(self._stored_input_shape)
            self._initialize_weights()

    # ── Weight helpers ─────────────────────────────────────────────────

    def _initialize_weights(self):
        """Create *n* weight vectors drawn from N(mean, std) then L2-normalise."""
        n = int(self.minicolumn_count)
        if self._input_shape is None:
            return

        h, w = self._input_shape
        size = h * w

        weights = np.random.normal(float(self.mean), float(self.std), (n, size))

        # L2-normalise each row so slerp operates on the unit sphere.
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self._weights = weights / norms

    def _build_grid_display(self) -> np.ndarray:
        """Arrange minicolumn weights into a square grid for visualisation."""
        n = int(self.minicolumn_count)
        k = int(np.sqrt(n))       # grid side length (assumes n is a perfect square)
        h, w = self._input_shape

        grid = np.zeros((k * h, k * w), dtype=np.float64)
        for i in range(n):
            row, col = divmod(i, k)
            weight_2d = self._weights[i].reshape(h, w)
            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = weight_2d
        return grid

    # ── Actions ────────────────────────────────────────────────────────

    def _on_reset_weights(self, params: dict):
        """Re-initialise all minicolumn weights from the current mean / std."""
        self._initialize_weights()
        if self._weights is not None:
            self.weights_display = self._build_grid_display()
        return {"status": "ok"}

    # ── Main loop ──────────────────────────────────────────────────────

    def process(self):
        if self.input is None:
            return

        input_data: np.ndarray = self.input

        if input_data.ndim != 2:
            self.info = f"Expected 2-D input, got {input_data.ndim}-D"
            return

        n = int(self.minicolumn_count)

        # (Re-)initialise weights when the input shape or minicolumn count
        # changes, or on the very first tick.
        needs_init = (
            self._weights is None
            or self._input_shape != input_data.shape
            or self._weights.shape[0] != n
        )
        if needs_init:
            self._input_shape = input_data.shape
            self._stored_input_shape = list(self._input_shape)
            self._initialize_weights()

        # Flatten and L2-normalise the input.
        input_flat = input_data.flatten().astype(np.float64)
        input_norm = np.linalg.norm(input_flat)
        if input_norm < 1e-10:
            self.info = "Input L2 norm ≈ 0 — skipping update"
            return
        input_normalized = input_flat / input_norm

        # ── Tangent-space attraction + repulsion update ────────────────
        lr = float(self.alpha)
        beta = float(self.beta)
        W = self._weights  # (n, D) — each row is a unit vector

        # Pre-compute all pairwise dot products: dots[i, j] = B_i · B_j
        dots_ww = W @ W.T  # (n, n)
        # Dot products of each weight with the input: dots_wa[i] = B_i · A
        dots_wa = W @ input_normalized  # (n,)

        new_weights = np.empty_like(W)

        for i in range(n):
            b_i = W[i]

            # Step 1 — Attraction toward input A
            #   t_attract = A - (A · B_i) * B_i  (project A onto tangent plane at B_i)
            t_attract = input_normalized - dots_wa[i] * b_i
            t_attract_norm = np.linalg.norm(t_attract)
            if t_attract_norm > 1e-10:
                t_attract /= t_attract_norm
            # else: A and B_i are (anti-)parallel; no attraction direction

            # Step 2 — Repulsion from other minicolumns B_j (uniform, no weighting)
            t_repel = np.zeros_like(b_i)
            for j in range(n):
                if j == i:
                    continue
                # Tangent vector from B_i toward B_j on the sphere
                t_ij = W[j] - dots_ww[i, j] * b_i
                t_ij_norm = np.linalg.norm(t_ij)
                if t_ij_norm < 1e-10:
                    continue  # B_j coincides with B_i or is antipodal
                t_ij /= t_ij_norm
                # Push *away* (negate the tangent toward B_j)
                t_repel -= t_ij

            # Normalise the summed repulsion direction
            t_repel_norm = np.linalg.norm(t_repel)
            if t_repel_norm > 1e-10:
                t_repel /= t_repel_norm

            # Step 3 — Combine in tangent space
            delta = lr * t_attract + beta * t_repel

            # Step 4 — Geodesic step: move and re-project onto sphere
            b_new = b_i + delta
            b_new_norm = np.linalg.norm(b_new)
            if b_new_norm > 1e-10:
                b_new /= b_new_norm
            new_weights[i] = b_new

        self._weights = new_weights

        # ── Output: dot product of raw input with each weight ──────────
        k = int(np.sqrt(n))
        # activations[i] = weight_i · input_flat  (raw, unnormalised)
        activations = self._weights @ input_flat  # (n,)
        self.output = activations.reshape(k, k)

        # ── Weights output: raw (n, D) weight matrix ──────────────────
        self.weights_out = self._weights.copy()

        # ── Visualise ──────────────────────────────────────────────────
        self.weights_display = self._build_grid_display()
        h, w = self._input_shape
        self.info = (
            f"{n} minicolumns ({k}×{k})  ·  "
            f"input {h}×{w}  ·  "
            f"display {k * h}×{k * w}"
        )


@dynamic
@node.custom("Cortex")
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

        W = self.weights          # (n, D)
        a = self.activations      # (k, k) or any shape with n total elements

        if W.ndim != 2:
            self.info = f"Weights must be 2-D, got {W.ndim}-D"
            return

        n, D = W.shape
        a_flat = a.flatten().astype(np.float64)  # (n,)

        if a_flat.shape[0] != n:
            self.info = (
                f"Shape mismatch: weights have {n} rows "
                f"but activations have {a_flat.shape[0]} elements"
            )
            return

        # x_reconstructed = W.T @ a  →  (D,)
        x_recon = W.T @ a_flat

        # Reshape to 2-D for output and display.
        # Infer the original input shape: try sqrt(D) for square inputs,
        # otherwise output as (1, D).
        side = int(np.sqrt(D))
        if side * side == D:
            x_recon_2d = x_recon.reshape(side, side)
        else:
            x_recon_2d = x_recon.reshape(1, D)

        self.output = x_recon_2d
        self.preview = x_recon_2d
        self.info = f"W ({n}×{D}) · a ({n},) → ({x_recon_2d.shape[0]}×{x_recon_2d.shape[1]})"
