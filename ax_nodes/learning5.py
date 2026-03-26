"""Contrast-Structure Hypothesis Learning (CSHL) on the unit hypersphere."""

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
class CSHL(Node):
    """Contrast-Structure Hypothesis Learning.

    Templates are competing whole-input hypotheses on the unit hypersphere.
    Activations are exact partial correlations (computed directly via G^{-1}).

    Each tick:
        1. L2-normalize input (assumed already mean-centered by LGN)
        2. Pearson correlations: c = W x_hat
        3. Regression coefficients: beta = G^{-1} c
        4. Partial correlations:
           a_i = beta_i / sqrt(G^{-1}_{ii} * (1 - c^T G^{-1} c))
        5. Learning: rotation toward/away from input, scaled by partial correlation
    """

    inputs = InputPort("Input", np.ndarray)

    amount = Integer("Amount", default=25)
    step_fraction = Range(
        "Step (η)", default=0.05, min_val=0.0, max_val=0.5, step=0.001
    )
    is_learning = Bool("Is Learning", default=True)

    weights = OutputPort("Weights", np.ndarray)
    raw_activations = OutputPort("Raw Activations", np.ndarray)
    activations = OutputPort("Activations", np.ndarray)

    w = State("Weights")

    weights_preview = Heatmap(
        "Weights", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual"
    )
    activations_preview = Heatmap(
        "Activations", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual"
    )
    info = Text("Info", default="idle")

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        self._input_shape = None
        self._dim = None
        self.w = None
    def _on_reset(self, _params=None):
        if self._dim is not None:
            self._init_weights(self._dim)
        return {"status": "ok"}

    def _init_weights(self, dim):
        self._dim = dim
        k = int(self.amount)
        w = np.random.randn(k, dim)
        # Mean-center each template, then L2-normalize
        w -= np.mean(w, axis=1, keepdims=True)
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self.w = w
        self.weights_preview = to_display_grid(self.w, patch_shape=self._input_shape)

    def process(self):
        eps = 1e-8

        if self.inputs is None:
            if self.w is not None:
                self.weights = self.w
            return

        raw_input = np.asarray(self.inputs, dtype=np.float64)
        if raw_input.ndim == 2:
            self._input_shape = raw_input.shape
        x = raw_input.ravel()

        # Lazy-init weights
        if self.w is None or self._dim != x.size:
            self._init_weights(x.size)

        w = np.asarray(self.w, dtype=np.float64)
        k = w.shape[0]

        # 1. L2-normalize input (already mean-centered by LGN)
        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self.weights = self.w
            self.activations = np.zeros(k)
            return
        x_hat = x / x_norm

        # 2. Exact partial correlations (direct solve)
        c = w @ x_hat                                   # (k,) Pearson correlations
        G = w @ w.T                                      # (k, k) Gram matrix
        G_inv = np.linalg.inv(G + eps * np.eye(k))      # regularized inverse
        beta = G_inv @ c                                 # regression coefficients
        residual_var = np.maximum(1.0 - np.dot(c, beta), eps)
        a = beta / np.sqrt(np.maximum(np.diag(G_inv), eps) * residual_var)

        # 3. Learning — rotation toward/away from input, scaled by partial correlation
        if self.is_learning:
            eta = float(self.step_fraction)

            # Tangent toward x_hat for each w_i
            dot = w @ x_hat  # (k,)
            tau = x_hat[None, :] - dot[:, None] * w  # (k, D)

            tau_norm = np.linalg.norm(tau, axis=1, keepdims=True)
            tau_hat = np.where(tau_norm > eps, tau / tau_norm, 0.0)

            # Rotation angle: scaled by partial correlation
            theta = (eta * a)[:, None]  # (k, 1)
            w_new = w * np.cos(theta) + tau_hat * np.sin(theta)

            # Renormalize + re-center
            w_new -= np.mean(w_new, axis=1, keepdims=True)
            w_new /= np.linalg.norm(w_new, axis=1, keepdims=True) + eps
            self.w = w_new

        # 4. Outputs
        self.weights = self.w
        self.raw_activations = c
        self.activations = a

        self.weights_preview = to_display_grid(self.w, patch_shape=self._input_shape)
        self.activations_preview = to_display_grid(a, patch_shape=self._input_shape)
        self.info = f"k={k}  dim={self._dim}  ||x||={x_norm:.3f}"


@branch("Hypercolumn/Conv")
class ConvCSHL(Node):
    """Convolutional CSHL — partial-correlation activations with shared weights.

    Tiles the input into non-overlapping PxP patches, computes partial
    correlations per patch using a single shared G^{-1}, and accumulates
    geodesic weight updates across all patches.

    Each tick:
        1. Tile input into non-overlapping patches
        2. Gram matrix inverse (once): G_inv = (W W^T + eps I)^{-1}
        3. Per patch (vectorized):
           c = W x_hat            (Pearson correlations)
           beta = G_inv c         (regression coefficients)
           a = beta / sqrt(diag(G_inv))  (partial correlations)
        4. Learning: accumulate tangents across patches, single rotation
    """

    input_data = InputPort("Input", np.ndarray)
    input_kernel_size = InputPort("Kernel Size", int)

    amount = Integer("Amount", default=25)
    step_fraction = Range(
        "Step (η)", default=0.05, min_val=0.0, max_val=0.5, step=0.001
    )
    is_learning = Bool("Is Learning", default=True)

    weights = OutputPort("Weights", np.ndarray)
    raw_activations = OutputPort("Raw Act", np.ndarray)
    activations = OutputPort("Activations", np.ndarray)
    output_kernel_size = OutputPort("Kernel Size", int)

    weights_conv_preview = Heatmap(
        "Weights", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual"
    )
    activations_conv_preview = Heatmap(
        "Activations", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual"
    )
    info = Text("Info", default="idle")

    w = State("Weights")

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        self.w = None
        self._dim = None

    def _on_reset(self, _params=None):
        if self._dim is not None:
            self._init_weights(self._dim)
        return {"status": "ok"}

    def _init_weights(self, dim):
        self._dim = dim
        k = int(self.amount)
        w = np.random.randn(k, dim)
        w -= np.mean(w, axis=1, keepdims=True)
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self.w = w

    def process(self):
        if self.input_data is None or self.input_kernel_size is None:
            if self.w is not None:
                self.weights = self.w
            return

        eps = 1e-8
        P = int(self.input_kernel_size)
        inp = np.asarray(self.input_data, dtype=np.float64)

        if inp.ndim == 1:
            side = int(np.sqrt(inp.size))
            inp = inp.reshape(side, side)

        if inp.ndim == 2:
            H, W, C = inp.shape[0], inp.shape[1], 1
        elif inp.ndim == 3:
            s = tuple(inp.shape)
            H, W, C = s[0], s[1], s[2]
        else:
            raise ValueError(f"ConvCSHL expects 2D or 3D input, got {inp.ndim}D")

        if H % P != 0 or W % P != 0:
            raise ValueError(
                f"ConvCSHL input ({H}, {W}) not divisible by kernel {P}"
            )
        grid_h = H // P
        grid_w = W // P
        dim = P * P * C

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

        # L2-normalize each patch (already mean-centered by upstream)
        patch_norms = np.linalg.norm(patches, axis=1, keepdims=True)
        valid = (patch_norms.ravel() > eps)
        X_hat = np.where(patch_norms > eps, patches / patch_norms, 0.0)

        # Gram matrix inverse (shared across all patches)
        G = w @ w.T
        G_inv = np.linalg.inv(G + eps * np.eye(k))
        G_inv_diag_sqrt = np.sqrt(np.maximum(np.diag(G_inv), eps))

        # Pearson correlations: (n_patches, k)
        C_raw = X_hat @ w.T

        # Partial correlations: (n_patches, k)
        beta = C_raw @ G_inv.T
        A = beta / G_inv_diag_sqrt[None, :]

        # Learning — update weights one patch at a time
        if self.is_learning:
            eta = float(self.step_fraction)

            for p in range(n_patches):
                if not valid[p]:
                    continue
                x_hat = X_hat[p]
                a = A[p]

                dot = w @ x_hat
                tau = x_hat[None, :] - dot[:, None] * w
                tau_norm = np.linalg.norm(tau, axis=1, keepdims=True)
                tau_hat = np.where(tau_norm > eps, tau / tau_norm, 0.0)

                theta = (eta * a)[:, None]
                w = w * np.cos(theta) + tau_hat * np.sin(theta)

            w -= np.mean(w, axis=1, keepdims=True)
            w /= np.linalg.norm(w, axis=1, keepdims=True) + eps
            self.w = w

        # Outputs as 3D tensors (grid_h, grid_w, k)
        self.weights = self.w
        self.raw_activations = C_raw.reshape(grid_h, grid_w, k)
        self.activations = A.reshape(grid_h, grid_w, k)
        self.output_kernel_size = P

        self.weights_conv_preview = to_display_grid(self.w)
        self.activations_conv_preview = to_display_grid(A)
        self.info = (
            f"kernel={P}x{P}  dim={dim}  "
            f"C={C}  "
            f"patches={n_patches} ({grid_h}x{grid_w})  "
            f"templates={k}  "
            f"out=({grid_h},{grid_w},{k})"
        )
