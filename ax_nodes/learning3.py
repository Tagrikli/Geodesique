"""Contrast-Residual Geodesic learning on the unit hypersphere."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Integer, Bool
from axonforge.core.descriptors.displays import Heatmap
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch
from .utilities import to_display_grid


@branch("Hypercolumn/Learning")
class ContrastResidualGeodesic(Node):
    """
    Contrast-Residual Geodesic learning on the unit hypersphere.

    Mean-centers the input to extract contrast (what distinguishes this pattern
    from the average), then uses per-template LOO residuals as the sole learning
    signal. No separate push/pull forces — templates learn toward unexplained
    features and pause when features are already claimed by others. The sphere
    constraint redistributes weight budget automatically.

    Preprocessing:
        x_c = x - mean(x)                              (mean-center)
        x_hat = x_c / ||x_c||                          (L2 normalize)

    Activations:
        s_i = w_i · x_hat                              (cosine similarity)
        a_i = max(s_i, 0)                              (raw activation)

    Reconstruction & LOO residual:
        x_recon = Σ_{j: a_j>0} a_j * w_j              (raw reconstruction)
        x_shadow_i = x_recon - a_i * w_i               (LOO shadow, unnormalized)
        x_shadow_hat_i = x_shadow_i / ||x_shadow_i||   (unit direction)
        r_i = x_hat - x_shadow_hat_i                   (pure directional residual)

    LOO-inhibited activation (uses unnormalized shadow):
        a_inh_i = max(s_i - w_i · x_shadow_i, 0)

    Learning (Rodrigues rotation with residual tangent):
        τ_i = r_i - (r_i · w_i) * w_i                 (project to tangent plane)
        θ_i = η * ||τ_i||                              (torque = rotation angle)
        w_i_new = w_i cos(θ_i) + τ̂_i sin(θ_i)        (geodesic step)

    Input Ports:
        inputs: Raw input pattern x (np.ndarray)

    Properties:
        amount: Number of templates/minicolumns
        dim: Dimensionality of each template
        step_fraction: Learning rate η (radians per unit torque)
        is_learning: Toggle learning on/off

    Output Ports:
        weights: Current weight matrix W (np.ndarray)
        raw_activations: ReLU activations a_i (np.ndarray)
        activations: LOO-inhibited activations (np.ndarray)
    """

    inputs = InputPort("Input", np.ndarray)

    amount = Integer("Amount", default=25)
    dim = Integer("Dim", default=784)
    step_fraction = Range(
        "Step (η)", default=0.05, min_val=0.0, max_val=0.5, step=0.001
    )
    trail_decay = Range(
        "Trail Decay (γ)", default=0.0, min_val=0.0, max_val=0.999, step=0.001, scale="log"
    )
    is_learning = Bool("Is Learning", default=True)

    weights = OutputPort("Weights", np.ndarray)
    raw_activations = OutputPort("Raw Act", np.ndarray)
    activations = OutputPort("Inh Act", np.ndarray)
    input_mean = OutputPort("Input Mean", np.ndarray)

    weights_preview = Heatmap("Weights", colormap="bwr", vmin=-1, vmax=1, scale_mode='manual')
    ema_input_preview = Heatmap("EMA Input", colormap="grayscale", vmin=0, vmax=1, scale_mode='manual')
    raw_activations_preview = Heatmap("Raw Act", colormap="grayscale", vmin=0, vmax=2, scale_mode='manual')

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        self._on_reset()

    def _on_reset(self, _params=None):
        w = np.random.randn(int(self.amount), int(self.dim))
        self._w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
        self._ema_buffer = None
        self.weights_preview = to_display_grid(self._w)
        return {"status": "ok"}

    def process(self):
        eps = 1e-8

        if self.inputs is None:
            self.weights = self._w
            return

        w = np.asarray(self._w, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()

        # Max-decay temporal integration (runs regardless of is_learning)
        gamma = float(self.trail_decay)
        if self._ema_buffer is None:
            self._ema_buffer = x.copy()
        else:
            self._ema_buffer = np.maximum(x, gamma * self._ema_buffer)
        x = self._ema_buffer
        self.ema_input_preview = to_display_grid(x)

        # Renormalize templates
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + eps)

        n_templates = w.shape[0]

        # 1. Mean-center and L2-normalize input
        x_mean = np.mean(x)
        x_c = x - x_mean
        x_c_norm = np.linalg.norm(x_c)
        if x_c_norm < eps:
            self.weights = self._w
            return
        x_hat = x_c / x_c_norm

        # 2. Activations
        s = w @ x_hat  # cosine similarities
        a = np.maximum(s, 0.0)  # raw ReLU activations

        # 3. Reconstruction (raw activations)
        x_recon = a @ w  # shape (D,)

        # 4. LOO shadow & residuals (both sides unit-normalized)
        # x_shadow_i = x_recon - a_i * w_i  (raw, for inhibition output)
        # x_shadow_hat_i = normalize(x_shadow_i)  (unit, for learning residual)
        x_shadow = x_recon[None, :] - (a[:, None] * w)  # (k, D)
        x_shadow_norm = np.linalg.norm(x_shadow, axis=1, keepdims=True)
        x_shadow_hat = x_shadow / (x_shadow_norm + eps)  # (k, D)

        # r_i = x_hat - x_shadow_hat_i  (pure directional residual)
        r_loos = x_hat[None, :] - x_shadow_hat  # (k, D)

        # 5. Learning
        if self.is_learning:
            eta = float(self.step_fraction)

            # Project residuals to tangent plane at each w_i
            r_dot_w = np.sum(r_loos * w, axis=1, keepdims=True)  # (k, 1)
            tau = r_loos - r_dot_w * w  # (k, D)

            tau_norm = np.linalg.norm(tau, axis=1, keepdims=True)  # (k, 1)
            tau_hat = np.where(tau_norm > eps, tau / tau_norm, 0.0)

            # Rodrigues rotation (active templates only)
            active = a > 0  # (k,)
            theta = eta * tau_norm  # (k, 1)
            w_rotated = w * np.cos(theta) + tau_hat * np.sin(theta)
            w_new = np.where(active[:, None], w_rotated, w)

            # Renormalize
            w_new = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
            self._w = w_new

        # 6. LOO-inhibited activations (unnormalized shadow, clamped to non-negative)
        shadow_overlap = np.maximum(np.sum(w * x_shadow, axis=1), 0.0)  # (k,)
        a_inh = np.maximum(s - shadow_overlap, 0.0)

        # 7. Output (scale by contrast norm)
        self.weights = self._w
        a_raw_out = a * x_c_norm
        a_inh_out = a_inh * x_c_norm
        self.raw_activations = a_raw_out
        self.activations = a_inh_out
        self.input_mean = np.array(x_mean)

        self.weights_preview = to_display_grid(self._w)
        self.raw_activations_preview = to_display_grid(a_raw_out)
