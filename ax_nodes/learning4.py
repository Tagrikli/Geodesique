"""Signed-Residual Geodesic learning on the unit hypersphere."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Integer, Bool
from axonforge.core.descriptors.displays import Heatmap
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch
from .utilities import to_display_grid


@branch("Hypercolumn/Learning")
class SignedResidualLearning(Node):
    """
    Signed-Residual Geodesic learning on the unit hypersphere.

    Fully signed variant — no ReLU anywhere. All templates participate in
    reconstruction and learning with their signed cosine similarity.

    Preprocessing:
        x_hat = x / ||x||                              (L2 normalize)

    Activations (signed, no ReLU):
        a_i = s_i = w_i · x_hat                        (cosine similarity, signed)

    Reconstruction (full signed):
        x_recon = Σ_j s_j * w_j                        (all templates, signed)

    LOO shadow:
        x_shadow_i = x_recon - s_i * w_i               (remove own contribution)
        x_shadow_hat_i = x_shadow_i / ||x_shadow_i||   (unit direction)

    Sign-aware residual:
        r_i = sign(s_i) · (x_hat - x_shadow_hat_i)     (flip for anti-aligned)

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
        raw_activations: Signed activations s_i (np.ndarray)
        activations: LOO-inhibited activations (np.ndarray)
    """

    inputs = InputPort("Input", np.ndarray)
    feedback = InputPort("Feedback", np.ndarray)

    amount = Integer("Amount", default=25)
    dim = Integer("Dim", default=784)
    step_fraction = Range(
        "Step (η)", default=0.05, min_val=0.0, max_val=0.5, step=0.001
    )
    gain_strength = Range(
        "Gain (γ)", default=0.5, min_val=0.0, max_val=50.0, step=0.01
    )
    trail_decay = Range(
        "Trail Decay (γ)", default=0.0, min_val=0.0, max_val=0.999, step=0.001, scale="log"
    )
    is_learning = Bool("Is Learning", default=True)

    weights = OutputPort("Weights", np.ndarray)
    raw_activations = OutputPort("Raw Act", np.ndarray)
    activations = OutputPort("Inh Act", np.ndarray)


    weights_preview = Heatmap("Weights", colormap="bwr", vmin=-1, vmax=1, scale_mode='manual')
    ema_input_preview = Heatmap("EMA Input", colormap="grayscale", vmin=0, vmax=1, scale_mode='manual')
    gain_input_preview = Heatmap("Gain Input", colormap="bwr", vmin=-1, vmax=1, scale_mode='manual')
    raw_activations_preview = Heatmap("Raw Act", colormap="bwr", vmin=-2, vmax=2, scale_mode='manual')

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

        # Feedback gain modulation (pre-EMA)
        if self.feedback is not None:
            fb = np.asarray(self.feedback, dtype=np.float64).ravel()
            g_raw = fb @ w                          # project to pixel space
            g_centered = g_raw - np.mean(g_raw)     # mean-center
            g_norm = np.linalg.norm(g_centered) + eps
            gamma = float(self.gain_strength)
            gain = 1.0 + gamma * (g_centered / g_norm)  # modulator centered at 1
            x = gain * x
            self.gain_input_preview = to_display_grid(x)

        # EMA temporal integration (runs regardless of is_learning)
        gamma = float(self.trail_decay)
        if self._ema_buffer is None:
            self._ema_buffer = x.copy()
        else:
            self._ema_buffer = (1.0 - gamma) * x + gamma * self._ema_buffer
        x = self._ema_buffer
        self.ema_input_preview = to_display_grid(x)

        # Renormalize templates
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + eps)

        # 1. L2-normalize input (assumed already mean-centered by LGN)
        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self.weights = self._w
            return
        x_hat = x / x_norm

        # 2. Activations (signed, no ReLU)
        s = w @ x_hat  # cosine similarities, signed

        # 3. Reconstruction (full signed)
        x_recon = s @ w  # shape (D,)

        # 4. LOO shadow & sign-aware residuals
        x_shadow = x_recon[None, :] - (s[:, None] * w)  # (k, D)
        x_shadow_norm = np.linalg.norm(x_shadow, axis=1, keepdims=True)
        x_shadow_hat = x_shadow / (x_shadow_norm + eps)  # (k, D)

        # r_i = sign(s_i) * (x_hat - x_shadow_hat_i)
        sign_s = np.sign(s)  # (k,)
        r_loos = sign_s[:, None] * (x_hat[None, :] - x_shadow_hat)  # (k, D)

        # 5. Learning (all templates, no active gate)
        if self.is_learning:
            eta = float(self.step_fraction)

            # Project residuals to tangent plane at each w_i
            r_dot_w = np.sum(r_loos * w, axis=1, keepdims=True)  # (k, 1)
            tau = r_loos - r_dot_w * w  # (k, D)

            tau_norm = np.linalg.norm(tau, axis=1, keepdims=True)  # (k, 1)
            tau_hat = np.where(tau_norm > eps, tau / tau_norm, 0.0)

            # Rodrigues rotation (all templates)
            theta = eta * tau_norm  # (k, 1)
            w_new = w * np.cos(theta) + tau_hat * np.sin(theta)

            # Renormalize
            w_new = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
            self._w = w_new

        # 6. LOO-inhibited activations
        shadow_overlap = np.sum(w * x_shadow, axis=1)  # (k,) — signed
        a_inh = s - shadow_overlap

        # 7. Output (scale by input norm)
        self.weights = self._w
        self.raw_activations = s * x_norm
        self.activations = a_inh * x_norm

        self.weights_preview = to_display_grid(self._w)
        self.raw_activations_preview = to_display_grid(s * x_norm)
