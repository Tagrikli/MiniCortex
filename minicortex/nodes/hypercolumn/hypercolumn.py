"""Cortical computation nodes for MiniCortex."""

import math
import numpy as np
from typing import Optional

from minicortex.core.node import Node
from minicortex.core.descriptors.ports import InputPort, OutputPort
from minicortex.core.descriptors.properties import Range, Integer, Bool
from minicortex.core.descriptors.displays import Vector2D, Text
from minicortex.core.descriptors.actions import Action
from minicortex.core.descriptors import branch


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


class HyperColumn(Node):

    # ── Ports ──────────────────────────────────────────────────────────
    i_input = InputPort("Input", np.ndarray)
    i_feedback = InputPort("Feedback", np.ndarray)
    o_output = OutputPort("Output", np.ndarray)
    o_weights = OutputPort("Weights", np.ndarray)

    # Props
    input_len = Integer("Input Size", default=28)
    minicolumn_count = Integer("Minicolumns", default=9)
    alpha = Range("Alpha", default=0.001, min_val=0.001, max_val=1, scale="log")
    beta = Range("Beta", default=0.05, min_val=0, max_val=5, scale="linear")
    gamma = Range("Gamma", default=0.1, min_val=0.5, max_val=1, scale="linear")
    feedback_ratio = Range("Feedback Ratio", default=0.0, min_val=0.0, max_val=1.0)

    is_learning = Bool("Learning")

    # Display
    weights_display = Vector2D("Weights", color_mode="bwr")
    delta_rot_display = Vector2D("Delta Rot")
    activation_raw = Vector2D("Activation Raw", color_mode="bwr")
    activation_final = Vector2D("Activation Final")
    error_map_display = Vector2D("Error Map", color_mode="bwr")
    reset_weights = Action("Reset Weights", callback="_on_reset_weights")
    log = Text("")

    def init(self):
        self.weights: Optional[np.ndarray] = self._initialize_weights()
        self.delta_rot = self.calculate_delta_rot_matrix()
        self.error_map = np.zeros(self.input_len**2)
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
        self.delta_rot = self.calculate_delta_rot_matrix()
        self.weights_display = to_display_grid(self.weights)
        self.delta_rot_display = self.delta_rot
        self.error_map = np.zeros(self.input_len**2)
        self.beta_per_neuron = np.full(self.minicolumn_count, self.beta)

        return {"status": "ok"}

    # ── Main loop ──────────────────────────────────────────────────────


    def process(self):
        if self.i_input is None:
            return

        ######## INPUT ########
        x: np.ndarray = self.i_input.flatten()
        norm = np.linalg.norm(x) + 1e-8
        x_norm = x / norm

        ######## FEEDFORWARD ########
        s_raw = x_norm @ self.weights.T

        s_final = (
            s_raw
            - (
                self.beta_per_neuron[:, None]
                * self.delta_rot
                * self.beta_per_neuron[None, :]
            )
            @ s_raw
        )

        ######## RECONSTRUCTION ########
        x_hat = self.weights.T @ s_final

        ######## ERROR TERMS ########

        # Pixel reconstruction error (world consistency)
        e_pix = x_norm - x_hat

        # Activation alignment error (hierarchical consistency)
        if self.i_feedback is not None:
            f = self.i_feedback

            # Normalize activation + feedback for structural alignment
            s_hat = s_final / (np.linalg.norm(s_final) + 1e-8)
            f_hat = f / (np.linalg.norm(f) + 1e-8)

            e_act = s_hat - f_hat  # activation-space directional error

            # Backproject to pixel space
            e_rep_pix = self.weights.T @ e_act
        else:
            e_rep_pix = np.zeros_like(e_pix)

        ######## MIXED ERROR ########

        error = (1-self.feedback_ratio) * e_pix + self.feedback_ratio * e_rep_pix


        error_norm = error / (np.linalg.norm(error) + 1e-8)

        ######## ADAPT INHIBITION (optional modulation) ########
        self.error_map = self.gamma * self.error_map + (1 - self.gamma) * error_norm
        error_map_norm = self.error_map / (np.linalg.norm(self.error_map) + 1e-8)

        error_alignment = self.weights @ error_map_norm
        error_alignment_norm = (error_alignment + 1) / 2
        self.beta_per_neuron = self.beta * (1 - error_alignment_norm) * 2

        ######## LEARNING (ROTATIONAL UPDATE) ########
        if self.is_learning:

            # projection of error onto weights
            dots = self.weights @ error_norm

            # perpendicular component
            e_perp = error_norm - dots[:, None] * self.weights

            norms = np.linalg.norm(e_perp, axis=1, keepdims=True) + 1e-8
            e_perp_hat = e_perp / norms

            # rotate proportional to activation magnitude
            theta = self.alpha * s_final

            self.weights = (
                np.cos(theta)[:, None] * self.weights
                + np.sin(theta)[:, None] * e_perp_hat
            )

            # re-normalize to stay on unit sphere
            self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + 1e-8

            self.delta_rot = self.calculate_delta_rot_matrix()

        ######## OUTPUTS ########
        self.o_output = s_final
        self.o_weights = self.weights

        ######## DISPLAY ########
        self.weights_display = to_display_grid(self.weights)
        self.delta_rot_display = self.delta_rot
        self.error_map_display = np.reshape(
            self.error_map, (self.input_len, self.input_len)
        )
        self.activation_raw = to_display_grid(e_pix)
        self.activation_final = to_display_grid(s_final)
        self.log = ""


    def calculate_delta_rot_matrix(self):
        dots = self.weights @ self.weights.T
        delta_rot = np.clip(dots, 0.0, 1.0)
        np.fill_diagonal(delta_rot, 0.0)
        return delta_rot

    def process2(self):
        if self.i_input is None:
            return

        ####### PROCESS ########

        x: np.ndarray = self.i_input.flatten()
        norm = np.linalg.norm(x) + 1e-8
        x_norm = x / norm

        # Calculate Activation
        s_raw = x_norm @ self.weights.T
        s_final = s_raw - self.beta * (self.delta_rot @ s_raw)  # (m,)
        s_final = np.maximum(0, s_final)

        # Reconstruct Input
        x_hat = norm * (self.weights.T @ s_final)

        # Calculate Coverage Gap
        error = x - x_hat

        # Weight update
        dW = np.outer(s_final, error)

        # Update Weights

        if self.is_learning:

            self.weights = self.weights + self.alpha * dW
            self.weights = self.weights / np.linalg.norm(
                self.weights, axis=1, keepdims=True
            )

            # Update Deltas for next iteration
            self.delta_rot = self.calculate_delta_rot_matrix()

        ####### Outputs #################

        self.o_output = s_final
        self.o_weights = self.weights

        ####### Log and display #########

        self.weights_display = to_display_grid(self.weights)
        self.delta_rot_display = self.delta_rot
        self.activation_raw = to_display_grid(s_raw)
        self.activation_final = to_display_grid(s_final)
        self.log = f""

    def get_weights_mosaic(self) -> np.ndarray:
        H, W = self.input_len, self.input_len
        M, D = self.weights.shape
        assert D == H * W

        # Smallest square side
        S = int(np.ceil(np.sqrt(M)))
        total = S * S
        pad = total - M

        weights = self.weights

        # Pad with zero-tiles if needed
        if pad > 0:
            weights = np.vstack([weights, np.zeros((pad, D), dtype=weights.dtype)])

        # Reshape into square mosaic
        mosaic = weights.reshape(S, S, H, W).transpose(0, 2, 1, 3).reshape(S * H, S * W)

        return mosaic


