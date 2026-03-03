"""Cortical computation nodes for MiniCortex."""

import math
import numpy as np
from typing import Optional

from minicortex.core.node import Node
from minicortex.core.descriptors.ports import InputPort, OutputPort
from minicortex.core.descriptors.properties import Range, Integer, Bool, Float
from minicortex.core.descriptors.displays import Vector2D, Text
from minicortex.core.descriptors.actions import Action
from minicortex.core.descriptors import branch

"""Lowercase Greek letters:

α (alpha) β (beta) γ (gamma) δ (delta) ε (epsilon) ζ (zeta) η (eta) θ (theta)

ι (iota) κ (kappa) λ (lambda) μ (mu) ν (nu) ξ (xi) ο (omicron) π (pi)

ρ (rho) σ (sigma) ς (final sigma) τ (tau) υ (upsilon) φ (phi) χ (chi) ψ (psi) ω (omega)
"""


def slerp_unit(u, v, alpha, eps=1e-8):
    # u, v are (approximately) unit vectors
    u = u / (np.linalg.norm(u) + eps)
    v = v / (np.linalg.norm(v) + eps)

    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-6:
        return u

    so = np.sin(omega)
    a = np.sin((1 - alpha) * omega) / so
    b = np.sin(alpha * omega) / so

    out = a * u + b * v
    return out / (np.linalg.norm(out) + eps)


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


class HyperColumnFeedbackDriven(Node):

    # ── Ports ──────────────────────────────────────────────────────────
    i_input = InputPort("Input", np.ndarray)
    i_feedback = InputPort("Feedback", np.ndarray)
    o_output = OutputPort("Output", np.ndarray)
    o_weights = OutputPort("Weights", np.ndarray)

    # Props
    input_len = Integer("Input Size", default=28)
    minicolumn_count = Integer("Minicolumns", default=9)

    alpha = Range("α", default=0.001, min_val=0.001, max_val=1, scale="log")
    beta = Range("β", default=0.001, min_val=0.001, max_val=10, scale="log")
    is_learning = Bool("Learning", default=True)

    # Display
    weights_display = Vector2D("Weights", color_mode="bwr")
    activation_raw = Vector2D("Activation Raw", color_mode="bwr")
    activation_final = Vector2D("Activation Final")
    reset_weights = Action("Reset Weights", callback="_on_reset_weights")
    log = Text("")

    def init(self):
        self.weights: Optional[np.ndarray] = self._initialize_weights()

    # ── Weight helpers ─────────────────────────────────────────────────
    def _initialize_weights(self):
        weights = np.random.randn(self.minicolumn_count, self.input_len**2)
        weights /= (np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8)
        return weights

    # ── Actions ────────────────────────────────────────────────────────
    def _on_reset_weights(self, params: dict):
        self.weights = self._initialize_weights()
        self.weights_display = to_display_grid(self.weights)
        return {"status": "ok"}

    # ── Main loop ──────────────────────────────────────────────────────
    def process(self):
        if self.i_input is None:
            return
        eps = 1e-8

        ######## INPUT ########
        x: np.ndarray = self.i_input.flatten()
        x_norm_factor = np.linalg.norm(x) + eps
        x_norm = x / x_norm_factor

        # keep unit templates
        self.weights = self.weights / (np.linalg.norm(self.weights, axis=1, keepdims=True) + eps)

        N = self.weights.shape[0]

        ######## FEEDBACK (signed) -> normalized match + separate scale ########
        # feedback can be negative/positive
        if self.i_feedback is None:
            # no feedback: neutral "all on"
            f = np.ones(N, dtype=self.weights.dtype)
            f_scale = 1.0
            f_hat = f
        else:
            f = self.i_feedback.flatten().astype(self.weights.dtype, copy=False)
            # global feedback magnitude (for stable normalization)
            f_scale = float(np.linalg.norm(f) + eps)
            f_hat = f / f_scale  # signed, L2-normalized feedback direction

        # "match" comes from normalized feedback; scale is applied only when used
        match = f_hat                      # signed match direction (requested)
        match_pos = np.maximum(match, 0.0) # attention/learning only for positive match

        ######## DELTA ROT (angular proximity) ########
        delta_rot = self.calculate_delta_rot_matrix()  # (N,N) in [0,1], diag=0

        ######## DENSITY (per-template crowding) ########
        density = delta_rot.sum(axis=1)                # (N,)
        density = density / (density.mean() + eps)     # stabilize across N

        ######## FEEDFORWARD ########
        s_raw = x_norm @ self.weights.T                # (N,)

        # Apply feedback match to expression:
        # - use normalized feedback direction (match_pos)
        # - apply magnitude scaling when used (f_scale)
        s_fb = s_raw * (match_pos * f_scale)

        # Lateral suppression ONLY when close (delta_rot) and ONLY from positive expressed activity
        inh = self.beta * (delta_rot @ np.maximum(s_fb, 0.0))
        s_final = s_fb - inh

        ######## LEARNING (feedback-driven; slowed by density; repels when crowded) ########
        if self.is_learning and (self.i_feedback is not None):
            # Attraction rotation magnitude:
            # - direction: feedback-normalized match (match_pos)
            # - scaled when used: multiply by f_scale
            # - slowed by density via beta
            theta_attr = self.alpha * (match_pos * f_scale) / (1.0 + self.beta * density)  # (N,)

            # Tangent direction toward x_norm on the unit sphere
            dots_x = (self.weights @ x_norm)[:, None]  # (N,1)
            tangent = x_norm[None, :] - dots_x * self.weights
            tangent_hat = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + eps)

            # Repulsion direction away from nearby templates (pure geometry)
            rep = delta_rot @ self.weights
            rep = rep - (np.sum(rep * self.weights, axis=1, keepdims=True) * self.weights)  # tangent proj
            rep_hat = rep / (np.linalg.norm(rep, axis=1, keepdims=True) + eps)

            # Repulsion magnitude:
            # - only affects positively matched (attended) neurons
            # - increases with crowding (density) and beta
            theta_rep = self.alpha * (match_pos * f_scale) * (self.beta * density)  # (N,)

            # Combine into one on-sphere rotation step
            t = (theta_attr[:, None] * tangent_hat) - (theta_rep[:, None] * rep_hat)
            t_norm = np.linalg.norm(t, axis=1, keepdims=True) + eps
            t_hat = t / t_norm
            theta = t_norm[:, 0]

            self.weights = np.cos(theta)[:, None] * self.weights + np.sin(theta)[:, None] * t_hat
            self.weights = self.weights / (np.linalg.norm(self.weights, axis=1, keepdims=True) + eps)

            # Recompute activations after learning (consistent output/display)
            s_raw = x_norm @ self.weights.T
            s_fb = s_raw * (match_pos * f_scale)
            inh = self.beta * (delta_rot @ np.maximum(s_fb, 0.0))
            s_final = s_fb - inh

        ######## OUTPUTS ########
        self.o_output = s_final * x_norm_factor
        self.o_weights = self.weights

        ######## DISPLAY ########
        self.weights_display = to_display_grid(self.weights)
        self.activation_raw = to_display_grid(s_raw)
        self.activation_final = to_display_grid(s_final)
        self.log = ""

    def calculate_delta_rot_matrix(self):
        dots = self.weights @ self.weights.T
        delta_rot = np.clip(dots, 0.0, 1.0)
        np.fill_diagonal(delta_rot, 0.0)
        return delta_rot

