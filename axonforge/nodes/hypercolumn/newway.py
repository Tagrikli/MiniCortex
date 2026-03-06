"""Cortical computation nodes for MiniCortex."""

import math
import numpy as np
from typing import Optional

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.properties import Range, Integer, Bool, Float
from axonforge.core.descriptors.displays import Vector2D, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch

"""Lowercase Greek letters:

α (alpha) β (beta) γ (gamma) δ (delta) ε (epsilon) ζ (zeta) η (eta) θ (theta)

ι (iota) κ (kappa) λ (lambda) μ (mu) ν (nu) ξ (xi) ο (omicron) π (pi)

ρ (rho) σ (sigma) ς (final sigma) τ (tau) υ (upsilon) φ (phi) χ (chi) ψ (psi) ω (omega)
"""

EPS = 1e-8


import numpy as np


class Hypercolumn_v3(Node):

    # ── Ports ──────────────────────────────────────────────────────────
    i_input: InputPort[np.ndarray] = InputPort("Input", np.ndarray)
    i_feedback: InputPort[np.ndarray] = InputPort("Feedback", np.ndarray)
    o_output = OutputPort("Output", np.ndarray)
    o_weights = OutputPort("Weights", np.ndarray)

    # ── Props ──────────────────────────────────────────────────────────
    input_len = Integer("Input Size", default=28)
    minicolumn_count = Integer("Minicolumns", default=9)

    alpha = Range("α", default=0.5, min_val=0.0001, max_val=20, scale="log")
    beta = Range("β", default=1.0, min_val=0.0, max_val=1, scale="linear")
    err_scale = Range("Error Scale", default=0.1, min_val=0.0, max_val=5.0)
    is_learning = Bool("Learning", default=True)
    feedback_strength = Range("Feedback Strength", default=0.5, min_val=0, max_val=1, step=0.01)

    # Density (pixel-space) dynamics
    dens_momentum = Range(
        "Density EMA λ", default=0.01, min_val=0.0001, max_val=0.5, scale="log"
    )

    # ── Display ────────────────────────────────────────────────────────
    log = Text("Log")
    weights_display = Vector2D("Weights", color_mode="bwr")
    activation_raw = Vector2D("Activation Raw", color_mode="bwr")
    activation_final = Vector2D("Activation Final", color_mode="bwr")
    density_display = Vector2D("Density (Pixel)", color_mode="bwr")
    error_display = Vector2D("Reconstruction Error", color_mode="bwr")
    reset = Action("Reset", callback="_on_reset")

    def init(self):
        self._on_reset({})

    # ── Actions ────────────────────────────────────────────────────────
    def _on_reset(self, params: dict):
        self.weights = np.random.randn(self.minicolumn_count, self.input_len**2)
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + 1e-8
        self.weights_display = to_display_grid(self.weights)
        self.dens_pix = np.full(self.input_len**2, 1.0, dtype=np.float32)
        return {"status": "ok"}

    def _calculate_entropy(self, a, eps):
        a_sum = np.sum(a) + eps
        p = a / a_sum
        h = -np.sum(p * np.log(p + eps)) / np.log(self.minicolumn_count + eps)
        return h

    def preprocess_input(self):
        x = self.i_input.flatten()
        x_norm_factor = np.linalg.norm(x) + EPS
        x_norm = x / x_norm_factor
        return x, x_norm_factor, x_norm

    def normalize_weights(self):
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + EPS

    def calculate_activations(self, x_norm):
        s_raw = self.weights @ x_norm
        suppression = self.weights @ self.weights.T  # neurons suppressing each other
        s_gated = s_raw - self.beta * (suppression @ s_raw)
        w_scaled = np.maximum(s_gated, 0.0)[:, None] * self.weights
        s_final = w_scaled @ x_norm
        return s_raw, s_final, None

    def reconstruction(self, s_final):
        return s_final @ self.weights

    def process(self):
        if self.i_input is None:
            return

        x, x_norm_factor, x_norm = self.preprocess_input()
        self.normalize_weights()

        s_raw, s_final, activations = self.calculate_activations(x_norm)
        recon = self.reconstruction(s_final)
        residual = self.reconstruction(s_raw - s_final)
        surprise_map = np.abs(residual * x_norm)

        ######## 5. LEARNING ########
        if self.is_learning:

            # Unsupervised: chase the residual — units naturally spread to cover gaps
            x_target = x_norm
            s_biased = self.weights @ x_target                         # (M,)

            # Annealing: dense areas freeze, sparse areas learn faster
            eff_alpha_i = self.alpha

            # Attraction tangent toward residual target
            t_attr = x_target[None, :] - s_biased[:, None] * self.weights
            t_attr_hat = t_attr / (np.linalg.norm(t_attr, axis=1, keepdims=True) + EPS)

            # Rotation magnitude — gated by how much unit already responds
            responsiveness = np.power(s_raw, 2)
            angle_to_target = np.arccos(np.clip(s_biased, -1.0, 1.0))
            theta = np.minimum(
                eff_alpha_i * angle_to_target * responsiveness,
                angle_to_target  # never overshoot
            )

            # Spherical update
            self.weights = (
                np.cos(theta)[:, None] * self.weights
                + np.sin(theta)[:, None] * t_attr_hat
            )
            self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + EPS


        ######## 6. OUTPUTS ########
        self.o_output = s_final * x_norm_factor
        self.o_weights = self.weights
        self.log = f"β: {self.beta:.3f}"
        self.weights_display = to_display_grid(self.weights)
        self.activation_final = to_display_grid(s_final)
        self.density_display = scale_to_bwr(self.dens_pix.reshape(self.input_len, self.input_len))
        self.error_display = surprise_map.reshape(self.input_len, self.input_len)


def to_display_grid(arr):
    """Convert 1D or 2D array to a square-ish 2D array for display.

    For 1D arrays: reshape into a near-square 2D array.
    For 2D arrays (weights matrix): create a mosaic of weight patches.
    """
    import numpy as np

    if arr.ndim == 1:
        n = len(arr)
        # Find dimensions for a near-square rectangle
        h = int(np.ceil(np.sqrt(n)))
        w = int(np.ceil(n / h))
        # Pad with zeros if needed
        padded = np.zeros(h * w, dtype=arr.dtype)
        padded[:n] = arr
        return padded.reshape(h, w)
    elif arr.ndim == 2:
        M, D = arr.shape
        # Check if D is a perfect square (for weight patches)
        sqrt_D = int(np.sqrt(D))
        if sqrt_D * sqrt_D == D:
            # This is a weight matrix with square patches
            H = sqrt_D
            W = sqrt_D
            S = int(np.ceil(np.sqrt(M)))
            total = S * S
            pad = total - M

            weights = arr
            if pad > 0:
                weights = np.vstack([weights, np.zeros((pad, D), dtype=weights.dtype)])

            mosaic = (
                weights.reshape(S, S, H, W).transpose(0, 2, 1, 3).reshape(S * H, S * W)
            )
            return mosaic
        else:
            # Not a square patch matrix, just reshape to near-square
            n = M * D
            h = int(np.ceil(np.sqrt(n)))
            w = int(np.ceil(n / h))
            flattened = arr.flatten()
            padded = np.zeros(h * w, dtype=arr.dtype)
            padded[:n] = flattened
            return padded.reshape(h, w)
    else:
        # Return as-is for other dimensions
        return arr

def scale_to_bwr(arr):
    """Scale array values to [-1, 1] range for full bwr color utilization.

    Uses min-max normalization: values are linearly mapped from [min, max]
    to [-1, 1]. This ensures unit vectors and other normalized data use
    the full blue-white-red color range instead of appearing whitish.
    """
    arr = np.asarray(arr)
    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        # Constant array - return zeros to show as white/neutral
        return np.zeros_like(arr)

    # Scale to [-1, 1]: 2 * (x - min) / (max - min) - 1
    return 2 * (arr - min_val) / (max_val - min_val) - 1
