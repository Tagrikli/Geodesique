"""Signed-Residual Geodesic learning on the unit hypersphere."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Integer, Bool
from axonforge.core.descriptors.displays import Heatmap, Text
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors.state import State
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


    w = State("Weights")

    weights_preview = Heatmap("Weights", colormap="bwr", vmin=-1, vmax=1, scale_mode='manual')
    ema_input_preview = Heatmap("EMA Input", colormap="grayscale", vmin=0, vmax=1, scale_mode='manual')
    gain_input_preview = Heatmap("Gain Input", colormap="bwr", vmin=-1, vmax=1, scale_mode='manual')
    raw_activations_preview = Heatmap("Raw Act", colormap="bwr", vmin=-2, vmax=2, scale_mode='manual')

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        self._on_reset()

    def _on_reset(self, _params=None):
        w = np.random.randn(int(self.amount), int(self.dim))
        self.w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
        self._ema_buffer = None
        self.weights_preview = to_display_grid(self.w)
        return {"status": "ok"}

    def process(self):
        eps = 1e-8

        if self.inputs is None:
            self.weights = self.w
            return

        w = np.asarray(self.w, dtype=np.float64)
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
            self.weights = self.w
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
            self.w = w_new

        # 6. LOO-inhibited activations
        shadow_overlap = np.sum(w * x_shadow, axis=1)  # (k,) — signed
        a_inh = s - shadow_overlap

        # 7. Output (scale by input norm)
        self.weights = self.w
        self.raw_activations = s * x_norm
        self.activations = a_inh * x_norm

        self.weights_preview = to_display_grid(self.w)
        self.raw_activations_preview = to_display_grid(s * x_norm)


@branch("Hypercolumn/Conv")
class ConvSRL(Node):
    """Convolutional Signed-Residual Geodesic learning with shared weights.

    Takes a full 2D array and kernel_size, tiles it into non-overlapping
    patches internally, applies SRL per patch with shared weights, and
    outputs a 2D activation map.  Weight updates (tangent vectors) are
    accumulated across all patches and applied as a single averaged
    Rodrigues rotation.

    dim is derived from kernel_size² — no manual dim field required.

    Input Ports:
        input_data:        2D array (image or upstream activation map)
        input_kernel_size: Kernel size (int)

    Properties:
        amount:        Number of templates (shared across patches)
        step_fraction: Learning rate η
        is_learning:   Toggle learning on/off

    Output Ports:
        weights:        Shared weight matrix (amount, dim)
        activations:    2D LOO-inhibited activation map
        raw_activations: 2D signed activation map
        output_kernel_size: Kernel size pass-through (int)
    """

    input_data = InputPort("Input", np.ndarray)
    input_kernel_size = InputPort("Kernel Size", int)

    amount = Integer("Amount", default=25)
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
    output_kernel_size = OutputPort("Kernel Size", int)

    weights_conv_preview = Heatmap(
        "Weights", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual"
    )
    activations_conv_preview = Heatmap(
        "Inh Act", colormap="bwr", vmin=-2, vmax=2, scale_mode="manual"
    )
    info = Text("Info", default="idle")

    w = State("Weights")

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        self.w = None
        self._dim = None
        self._ema_buffer = None

    def _on_reset(self, _params=None):
        if self._dim is not None:
            self._init_weights(self._dim)
        self._ema_buffer = None
        return {"status": "ok"}

    def _init_weights(self, dim):
        self._dim = dim
        k = int(self.amount)
        w = np.random.randn(k, dim)
        self.w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)

    def process(self):
        if self.input_data is None or self.input_kernel_size is None:
            if self.w is not None:
                self.weights = self.w
            return

        eps = 1e-8
        P = int(self.input_kernel_size)

        inp = np.asarray(self.input_data, dtype=np.float64)

        # EMA temporal integration
        gamma = float(self.trail_decay)
        if self._ema_buffer is None or self._ema_buffer.shape != inp.shape:
            self._ema_buffer = inp.copy()
        else:
            self._ema_buffer = (1.0 - gamma) * inp + gamma * self._ema_buffer
        inp = self._ema_buffer

        if inp.ndim == 1:
            side = int(np.sqrt(inp.size))
            inp = inp.reshape(side, side)

        # Handle both 2D (H, W) and 3D (H, W, C) inputs
        if inp.ndim == 2:
            H, W, C = inp.shape[0], inp.shape[1], 1
        elif inp.ndim == 3:
            s = tuple(inp.shape)
            H, W, C = s[0], s[1], s[2]  # type: ignore[index]
        else:
            raise ValueError(f"ConvSRL expects 2D or 3D input, got {inp.ndim}D")

        if H % P != 0 or W % P != 0:
            raise ValueError(
                f"ConvSRL input spatial shape ({H}, {W}) is not divisible by kernel size {P}"
            )
        grid_h = H // P
        grid_w = W // P
        dim = P * P * C

        # Auto-init or re-init weights if dim changed
        if self.w is None or self._dim != dim:
            self._init_weights(dim)

        # Tile into non-overlapping patches: (n_patches, dim)
        if C == 1:
            if inp.ndim == 3:
                inp = inp[:, :, 0]
            patches = (
                inp.reshape(grid_h, P, grid_w, P)
                .transpose(0, 2, 1, 3)
                .reshape(grid_h * grid_w, P * P)
            )
        else:
            patches = (
                inp.reshape(grid_h, P, grid_w, P, C)
                .transpose(0, 2, 1, 3, 4)
                .reshape(grid_h * grid_w, P * P * C)
            )
        n_patches = patches.shape[0]

        w = np.asarray(self.w, dtype=np.float64)
        k = w.shape[0]

        # Renormalize templates
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + eps)

        tau_accum = np.zeros_like(w)
        all_raw = np.zeros((n_patches, k))
        all_inh = np.zeros((n_patches, k))
        valid_count = 0

        for p in range(n_patches):
            x = patches[p]

            # L2-normalize (patch already mean-centered by ConvLGN)
            x_norm = np.linalg.norm(x)
            if x_norm < eps:
                continue
            x_hat = x / x_norm

            # Signed activations
            s = w @ x_hat

            # Full signed reconstruction
            x_recon = s @ w

            # LOO shadow
            x_shadow = x_recon[None, :] - (s[:, None] * w)
            x_shadow_norm = np.linalg.norm(x_shadow, axis=1, keepdims=True)
            x_shadow_hat = x_shadow / (x_shadow_norm + eps)

            # Sign-aware residuals
            sign_s = np.sign(s)
            r_loos = sign_s[:, None] * (x_hat[None, :] - x_shadow_hat)

            # Tangent projection
            r_dot_w = np.sum(r_loos * w, axis=1, keepdims=True)
            tau = r_loos - r_dot_w * w
            tau_accum += tau
            valid_count += 1

            # LOO-inhibited activations
            shadow_overlap = np.sum(w * x_shadow, axis=1)
            all_raw[p] = s * x_norm
            all_inh[p] = (s - shadow_overlap) * x_norm

        # Rodrigues rotation with averaged tangent
        if self.is_learning and valid_count > 0:
            tau_avg = tau_accum / valid_count
            tau_norm = np.linalg.norm(tau_avg, axis=1, keepdims=True)
            tau_hat = np.where(tau_norm > eps, tau_avg / tau_norm, 0.0)

            eta = float(self.step_fraction)
            theta = eta * tau_norm
            w_new = w * np.cos(theta) + tau_hat * np.sin(theta)
            w_new = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
            self.w = w_new

        # Output as 3D tensors (grid_h, grid_w, k)
        inh_3d = all_inh.reshape(grid_h, grid_w, k)
        raw_3d = all_raw.reshape(grid_h, grid_w, k)

        self.weights = self.w
        self.activations = inh_3d
        self.raw_activations = raw_3d
        self.output_kernel_size = P

        # Display previews (2D mosaics for visualization only)
        self.weights_conv_preview = to_display_grid(self.w)
        self.activations_conv_preview = to_display_grid(all_inh)

        self.info = (
            f"kernel={P}x{P}  dim={dim}  "
            f"C={C}  "
            f"patches={n_patches} ({grid_h}x{grid_w})  "
            f"templates={k}  "
            f"W=({k},{dim})  "
            f"out=({grid_h},{grid_w},{k})"
        )
