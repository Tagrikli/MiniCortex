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


class HyperColumnAttn(Node):

    # ── Ports ──────────────────────────────────────────────────────────
    i_input: InputPort[np.ndarray] = InputPort("Input", np.ndarray)
    i_feedback: InputPort[np.ndarray] = InputPort("Feedback", np.ndarray)
    o_output = OutputPort("Output", np.ndarray)
    o_weights = OutputPort("Weights", np.ndarray)

    # ── Props ──────────────────────────────────────────────────────────
    input_len = Integer("Input Size", default=28)
    minicolumn_count = Integer("Minicolumns", default=9)

    alpha = Range("α", default=0.5, min_val=0.0001, max_val=20, scale="log")
    beta = Range("β", default=1.0, min_val=0, max_val=30, scale="linear")
    err_scale = Range("Error Scale", default=0.1, min_val=0.0, max_val=5.0)
    is_learning = Bool("Learning", default=True)
    feedback_ratio = Range("FR", default=0.5, min_val=0, max_val=1, step=0.01)

    # Density (pixel-space) dynamics
    dens_momentum = Range(
        "Density EMA λ", default=0.01, min_val=0.0001, max_val=0.5, scale="log"
    )

    # ── Display ────────────────────────────────────────────────────────

    log = Text("Beta")
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
        # Normalized Shannon Entropy (0.0 to 1.0)
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
        # 1) nonnegative evidence (match to input)
        s_raw = self.weights @ x_norm  # (M,)
        s_pos = np.maximum(s_raw, 0.0)  # (M,)  <-- no negatives

        # 2) template-to-template similarity (how much they compete)
        S = self.weights @ self.weights.T  # (M,M)
        np.fill_diagonal(S, 0.0)  # IMPORTANT: no self-suppression

        # optional: only allow inhibitory connections to be nonnegative
        # (anti-correlated templates shouldn't "help" via negative inhibition)
        S = np.maximum(S, 0.0)

        # 3) inhibition driven by who is active
        k = 1#self.beta  # use beta as inhibition strength (or any scalar)
        inhib = k * (S @ s_pos)  # (M,)

        # 4) subtract inhibition and rectify
        s_final = np.maximum(s_pos - inhib, 0.0)  # (M,)  <-- guaranteed nonnegative
        return s_raw, s_final

    def reconstruction(self, s_final):
        return s_final @ self.weights

    def update_density_map(self, density_map_target):
        self.dens_pix = (
            1.0 - self.dens_momentum
        ) * self.dens_pix + self.dens_momentum * (density_map_target * self.err_scale)

        self.dens_pix /= np.mean(self.dens_pix) + EPS

    def calculate_density_i(self):
        w_mask = np.abs(self.weights)
        w_mask /= np.sum(w_mask, axis=1, keepdims=True) + EPS
        dens_i = w_mask @ self.dens_pix
        return dens_i

    def process(self):
        if self.i_input is None:
            return

        x, x_norm_factor, x_norm = self.preprocess_input()
        self.normalize_weights()

        density_i = self.calculate_density_i()
        s_raw, s_final = self.calculate_activations(x_norm)
        reconstruction = self.reconstruction(s_final)
        residual = x_norm - reconstruction
        surprise_map = np.abs(residual * x_norm)

        # self.update_density_map(surprise_map)

        ######## 5. LEARNING (DENSITY GRAVITY & ANNEALING) ########
        if self.is_learning:

            # Shift the learning target toward pixels with high 'Surprise'
            x_biased = x_norm  # * self.dens_pix
            x_biased_norm = x_biased / (np.linalg.norm(x_biased) + EPS)
            s_biased = self.weights @ x_biased_norm

            ### HEEEYYYYY ####
            if self.i_feedback is not None:
                x_biased = (
                    1 - self.feedback_ratio
                ) * x_biased_norm + self.feedback_ratio * (
                    self.weights.T @ self.i_feedback
                )
                x_biased_norm = x_biased / (np.linalg.norm(x_biased) + EPS)
                s_biased = self.weights @ x_biased_norm
            ###

            # Annealing: Reduce alpha/beta in dense areas to "freeze" details
            scale_factor_i = 1.0 / (density_i + EPS)
            eff_alpha_i = self.alpha * scale_factor_i
            eff_beta_i = self.beta * scale_factor_i

            # A. Attraction toward Biased Target
            t_attr = x_biased_norm[None, :] - s_biased[:, None] * self.weights
            t_attr_hat = t_attr / (np.linalg.norm(t_attr, axis=1, keepdims=True) + EPS)

            # B. Repulsion (Structural Orthogonalization)
            proximity = self.weights @ self.weights.T
            np.fill_diagonal(proximity, 0.0)
            t_rep_raw = proximity @ self.weights
            dots_rep = np.sum(t_rep_raw * self.weights, axis=1, keepdims=True)
            t_rep = t_rep_raw - dots_rep * self.weights
            t_rep_hat = t_rep / (np.linalg.norm(t_rep, axis=1, keepdims=True) + EPS)

            # C. Net Goal & Rotation
            v_net = t_attr_hat - (eff_beta_i[:, None] * t_rep_hat)
            dir_tan = v_net / (np.linalg.norm(v_net, axis=1, keepdims=True) + EPS)

            # D. Rotation Magnitude (Quadratic response to input match)
            responsiveness = np.power(s_raw, 2)
            angle_to_target = np.arccos(np.clip(s_biased, -1.0, 1.0))
            theta = eff_alpha_i * angle_to_target * responsiveness

            # E. Spherical Update
            self.weights = (
                np.cos(theta)[:, None] * self.weights + np.sin(theta)[:, None] * dir_tan
            )
            self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + EPS

        ######## 6. OUTPUTS ########
        self.o_output = s_final * x_norm_factor
        self.o_weights = self.weights
        self.log = f"β: {self.beta:.3f}"
        self.weights_display = to_display_grid(self.weights)
        self.activation_final = to_display_grid(s_final)
        self.density_display = scale_to_bwr(
            self.dens_pix.reshape(self.input_len, self.input_len)
        )
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
