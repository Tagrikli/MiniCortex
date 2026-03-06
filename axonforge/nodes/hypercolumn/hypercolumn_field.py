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


class HyperColumnFieldDriven(Node):

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
    feedback_ratio = Range("Feedback Ratio", default=0.0, min_val=0.0, max_val=1.0)
    is_learning = Bool("Learning", default=True)

    # Display
    weights_display = Vector2D("Weights", color_mode="bwr")
    display_pix = Vector2D("PIX Map", color_mode="bwr")
    display_rep = Vector2D("REP Map", color_mode="bwr")
    activation_raw = Vector2D("Activation Raw", color_mode="bwr")
    activation_final = Vector2D("Activation Final")
    reset_weights = Action("Reset Weights", callback="_on_reset_weights")
    log = Text("")

    def init(self):
        self.weights: Optional[np.ndarray] = self._initialize_weights()
        self.map_pix = np.zeros(self.input_len**2)
        self.map_rep = np.zeros(self.input_len**2)
        self.beta_per_neuron = np.full(self.minicolumn_count, self.beta)

    # ── Weight helpers ─────────────────────────────────────────────────

    def _initialize_weights(self):

        weights = np.random.randn(self.minicolumn_count, self.input_len**2)
        norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
        weights = weights / norms
        return weights

    # ── Actions ────────────────────────────────────────────────────────

    def _on_reset_weights(self, params: dict):
        self.weights = self._initialize_weights()
        self.map_pix = np.zeros(self.input_len**2)
        self.map_rep = np.zeros(self.input_len**2)
        self.beta_per_neuron = np.full(self.minicolumn_count, self.beta)

        self.weights_display = to_display_grid(self.weights)
        self.display_pix = to_display_grid(self.map_pix)
        self.display_rep = to_display_grid(self.map_rep)

        return {"status": "ok"}

    # ── Main loop ──────────────────────────────────────────────────────

    def process(self):
        if self.i_input is None:
            return

        eps = 1e-8
        rho = self.feedback_ratio

        ######## INPUT ########
        x: np.ndarray = self.i_input.flatten()
        x_norm_factor = np.linalg.norm(x) + eps
        x_norm = x / x_norm_factor

        ######## FEEDFORWARD ########
        

        s_raw = x_norm @ self.weights.T
        gate = 1.0 - self.beta * self.beta_per_neuron
        s_final = s_raw * gate

        ######## RECONSTRUCTION ########
        x_hat = self.weights.T @ s_final

        ######## ERROR TERMS ########
        # --- World consistency ---
        e_pix = x_norm - x_hat

        # --- Hierarchical consistency (direction-only in activation space) ---
        if self.i_feedback is not None:
            f = self.i_feedback

            s_hat = s_final / (np.linalg.norm(s_final) + eps)
            f_hat = f / (np.linalg.norm(f) + eps)

            delta = f_hat - s_hat
        else:
            delta = np.zeros_like(s_final)

        ######## STORE MAPS ########
        self.map_pix = e_pix
        self.map_rep = delta

        ######## MIX ROTATIONS (TANGENT SPACE) ########
        # Pixel space tangent for reconstruction
        e_pix_hat = self.map_pix / (np.linalg.norm(self.map_pix) + eps)
        dots_pix = self.weights @ e_pix_hat
        t_pix = e_pix_hat - dots_pix[:, None] * self.weights
        t_pix_hat = t_pix / (np.linalg.norm(t_pix, axis=1, keepdims=True) + eps)

        # Activation space tangent for feedback (rotate toward/away from x_norm)
        dots_x = self.weights @ x_norm
        t_fb = x_norm - dots_x[:, None] * self.weights
        t_fb_hat = t_fb / (np.linalg.norm(t_fb, axis=1, keepdims=True) + eps)

        # Feedback magnitude from activation mismatch
        e_rep_mag = np.linalg.norm(delta)

        # Mix
        t_mix = (1.0 - rho) * t_pix_hat + rho * delta[:, None] * t_fb_hat
        t_mix_hat = t_mix / (np.linalg.norm(t_mix, axis=1, keepdims=True) + eps)

        ######## ADAPT INHIBITION ########
        a_pix = np.clip(dots_pix, -1.0, 1.0)
        a_fb = np.clip(delta, -1.0, 1.0)
        a = (1.0 - rho) * a_pix + rho * a_fb

        ######## LEARNING (ROTATIONAL UPDATE) ########
        if self.is_learning:
            self.beta_per_neuron = 0.5 * (1.0 - a)

            theta = self.alpha * self.beta_per_neuron * np.maximum(s_final, 0)

            self.weights = (
                np.cos(theta)[:, None] * self.weights
                + np.sin(theta)[:, None] * t_mix_hat
            )

            self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + eps

        ######## OUTPUTS ########
        self.o_output = s_final * x_norm_factor
        self.o_weights = self.weights

        ######## DISPLAY ########
        self.weights_display = to_display_grid(self.weights)
        self.display_pix = to_display_grid(self.map_pix)
        self.display_rep = to_display_grid(self.map_rep)

        self.activation_raw = to_display_grid(s_raw)
        self.activation_final = to_display_grid(s_final)
        self.log = ""

