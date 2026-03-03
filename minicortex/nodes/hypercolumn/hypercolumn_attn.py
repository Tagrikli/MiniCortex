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


import numpy as np

# Assumes you already have:
# - Node, InputPort, OutputPort
# - Integer, Range, Bool, Vector2D, Action, Text
# - to_display_grid(...)
#
# This rewrite implements:
# 1) Pixel-space density map dens_pix >= 0 as EMA of a goal signal (default: squared recon error)
# 2) Templates are attracted toward dense areas by weighting the learning target with dens_pix^p
# 3) Rotation is slowed in dense areas (divide step by dens_unit)
# 4) Inhibition is LOWER in dense areas (beta_eff = beta / dens_unit)
# 5) Only relevant templates learn (gated by positive activation AND positive error-alignment)


class HyperColumnAttn(Node):

    # ── Ports ──────────────────────────────────────────────────────────
    i_input = InputPort("Input", np.ndarray)
    o_output = OutputPort("Output", np.ndarray)
    o_weights = OutputPort("Weights", np.ndarray)

    # ── Props ──────────────────────────────────────────────────────────
    input_len = Integer("Input Size", default=28)
    minicolumn_count = Integer("Minicolumns", default=9)

    alpha = Range("α", default=0.001, min_val=0.0001, max_val=20, scale="log")
    #beta = Range("β", default=0.1, min_val=0.0001, max_val=500, scale="log")
    err_scale = Range("Error Scale", default=0.1, min_val=0.0, max_val=5.0)
    is_learning = Bool("Learning", default=True)

    # Density (pixel-space) dynamics
    dens_momentum = Range(
        "Density EMA λ", default=0.01, min_val=0.0001, max_val=0.5, scale="log"
    )
    beta_momentum = Range(
        "Beta λ", default=0.01, min_val=0.0001, max_val=1, scale="log"
    )

    # Target entropy for homeostasis (the "personality" of the column)
    # 0.1 = Extremely sparse (1 winner), 0.5 = Healthy minimal set (2-3 winners), 0.9 = Dense redundant
    target_entropy = Range(
        "Target Entropy", default=0.4, min_val=0.01, max_val=0.99, step=0.01, scale="linear"
    )

    # Manual beta adjustment controls
    manual_beta = Bool("Manual Beta", default=False)
    manual_beta_value = Range(
        "Manual β Value", default=1.0, min_val=0.5, max_val=80, scale="linear"
    )


    # ── Display ────────────────────────────────────────────────────────

    log = Text("Beta")
    weights_display = Vector2D("Weights", color_mode="bwr")
    activation_raw = Vector2D("Activation Raw", color_mode="bwr")
    activation_final = Vector2D("Activation Final", color_mode="bwr")
    density_display = Vector2D("Density (Pixel)", color_mode="bwr")
    error_display = Vector2D("Reconstruction Error", color_mode="bwr")
    reset_weights = Action("Reset Weights", callback="_on_reset_weights")

    def init(self):
        self.weights: np.ndarray = self._initialize_weights()
        self.dens_pix = np.full(self.input_len**2, 1.0)
        self.beta = 1.0


    # ── Weight helpers ─────────────────────────────────────────────────
    def _initialize_weights(self) -> np.ndarray:
        # Non-negative init (MNIST-like); if your inputs can be signed, switch to randn.
        W = np.abs(np.random.randn(self.minicolumn_count, self.input_len**2)).astype(
            np.float32
        )
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
        return W

    # ── Actions ────────────────────────────────────────────────────────
    def _on_reset_weights(self, params: dict):
        self.weights = self._initialize_weights()
        self.weights_display = to_display_grid(self.weights)
        self.dens_pix = np.full(self.input_len**2, 1.0, dtype=np.float32)
        self.beta = 10.0

        # Reset PID controller state
        self._pid_integral = 0.0
        self._pid_prev_error = 0.0
        self._pid_initialized = False

        return {"status": "ok"}

    def _calculate_entropy(self, a, eps):
            a_sum = np.sum(a) + eps
            p = a / a_sum
            # Normalized Shannon Entropy (0.0 to 1.0)
            h = -np.sum(p * np.log(p + eps)) / np.log(self.minicolumn_count + eps)
            return h

    def process(self):
        if self.i_input is None:
            return

        eps = 1e-8
        N, D = self.weights.shape

        ######## 1. INPUT & PRE-PROCESSING ########
        x: np.ndarray = self.i_input.flatten().astype(np.float32)
        x_norm_factor = float(np.linalg.norm(x) + eps)
        x_norm = x / x_norm_factor 

        # Weights re-normalization for safety
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + eps

        ######## 2. INFERENCE (ENTROPY-DRIVEN INHIBITION) ########
        # Initial feedforward pass
        s_raw = self.weights @ x_norm  
        a = np.maximum(s_raw, 0.0)      

        # Dynamic Beta calculation based on the Importance Map (Annealing)
        # High local importance (complexity) lowers inhibition to allow collaboration
        local_importance = np.dot(x_norm, self.dens_pix)
        effective_beta_fb = self.beta / (local_importance + eps)
        
        gain = 1.0 / (1.0 + effective_beta_fb)
        s_final = a * gain

        ######## 3. SURPRISE & DENSITY DYNAMICS ########
        # Reconstruction: How well do the active neurons explain the input?
        reconstruction = s_final @ self.weights 
        residual = x_norm - reconstruction
        
        # Surprise Logic: We filter the residual by the input signal.
        # This prevents the density map from 'chasing' high-dimensional noise.
        # It only cares about errors where there is actual input content.
        surprise_map = np.abs(residual * x_norm)
        
        # Update Density (EMA)
        self.dens_pix = (1.0 - self.dens_momentum) * self.dens_pix + \
                        self.dens_momentum * (surprise_map * self.err_scale)
        
        # Keep global density average at 1.0 (Conservation of Attention)
        self.dens_pix /= (np.mean(self.dens_pix) + eps)

        ######## 4. HOMEOTASIS (BETA ADAPTATION) ########
        h_current = self._calculate_entropy(a, eps) # Use pre-gain 'a' for competition state
        
        if self.manual_beta:
            # Use manually specified beta value from slider
            self.beta = self.manual_beta_value
        else:
            # Automatic beta adaptation based on entropy
            # If entropy (representation overlap) is high, increase beta
            # If entropy (sparsity) is too low, decrease beta
            entropy_error = h_current - self.target_entropy
            self.beta += self.beta_momentum * entropy_error
            self.beta = np.clip(self.beta, 0.01, self.minicolumn_count)

        ######## 5. LEARNING (DENSITY GRAVITY & ANNEALING) ########
        if self.is_learning:
            # Shift the learning target toward pixels with high 'Surprise'
            x_biased = x_norm * self.dens_pix
            x_biased_norm = x_biased / (np.linalg.norm(x_biased) + eps)
            s_biased = self.weights @ x_biased_norm

            # Annealing: Reduce alpha/beta in dense areas to "freeze" details
            scale_factor = 1.0 / (local_importance + eps)
            eff_alpha = self.alpha * scale_factor
            eff_beta = self.beta * scale_factor

            # A. Attraction toward Biased Target
            t_attr = x_biased_norm[None, :] - s_biased[:, None] * self.weights
            t_attr_hat = t_attr / (np.linalg.norm(t_attr, axis=1, keepdims=True) + eps)

            # B. Repulsion (Structural Orthogonalization)
            proximity = self.weights @ self.weights.T
            np.fill_diagonal(proximity, 0.0)
            t_rep_raw = proximity @ self.weights 
            dots_rep = np.sum(t_rep_raw * self.weights, axis=1, keepdims=True)
            t_rep = t_rep_raw - dots_rep * self.weights
            t_rep_hat = t_rep / (np.linalg.norm(t_rep, axis=1, keepdims=True) + eps)

            # C. Net Goal & Rotation
            v_net = t_attr_hat - (eff_beta * t_rep_hat)
            dir_tan = v_net / (np.linalg.norm(v_net, axis=1, keepdims=True) + eps)

            # D. Rotation Magnitude (Quadratic response to input match)
            responsiveness = np.power(a, 2)
            angle_to_target = np.arccos(np.clip(s_biased, -1.0, 1.0))
            theta = eff_alpha * angle_to_target * responsiveness

            # E. Spherical Update
            self.weights = (np.cos(theta)[:, None] * self.weights + 
                            np.sin(theta)[:, None] * dir_tan)
            self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + eps

        ######## 6. OUTPUTS ########
        self.o_output = s_final * x_norm_factor
        self.o_weights = self.weights
        self.log = f"β: {self.beta:.3f} | H: {h_current:.2f} | Imp: {local_importance:.2f}"
        
        self.weights_display = to_display_grid(self.weights)
        self.activation_final = to_display_grid(s_final)
        self.density_display = self.dens_pix.reshape(self.input_len, self.input_len)
        self.error_display = surprise_map.reshape(self.input_len, self.input_len) 



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
