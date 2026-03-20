import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.displays import Heatmap
from .utilities import to_display_grid


@branch("Hypercolumn")
class Reconstruction(Node):
    weights = InputPort("Weights", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    input_mean = InputPort("Input Mean", float)

    result = OutputPort("Result", np.ndarray)

    result_heatmap = Heatmap(
        "Result Heatmap",
        colormap="grayscale",
        scale_mode="manual",
        vmin=0.0,
        vmax=1.0,
    )

    def process(self):
        if self.weights is None or self.activations is None:
            return

        result = self.weights.T @ self.activations

        if self.input_mean is not None:
            result = result + self.input_mean

        self.result = result
        self.result_heatmap = to_display_grid(result)


@branch("Hypercolumn/Conv")
class ConvReconstruction(Node):
    """Reconstruct from convolutional SRL activations and shared weights.

    Takes the 3D activation tensor (grid_h, grid_w, K) and shared weight
    matrix from ConvSRL, reconstructs each patch via activations @ weights,
    adds back per-patch means if provided, and assembles the result.

    Output is 2D (grid_h*P, grid_w*P) when the layer operated on a 2D input,
    or 3D (grid_h*P, grid_w*P, C) when it operated on a multi-channel input.
    Channel count is inferred from weights: C = dim // (P*P).

    Input Ports:
        weights:           Shared weight matrix (K, dim)
        activations:       3D activation tensor (grid_h, grid_w, K) from ConvSRL
        input_kernel_size: Kernel size (int)
        input_means:       Per-patch means from ConvLGN (n_patches,), optional

    Output Ports:
        result: Reconstructed array (2D or 3D depending on channel count)
    """

    weights = InputPort("Weights", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    input_kernel_size = InputPort("Kernel Size", int)
    input_means = InputPort("Patch Means", np.ndarray)

    result = OutputPort("Result", np.ndarray)

    result_heatmap = Heatmap(
        "Result Heatmap",
        colormap="grayscale",
        scale_mode="manual",
        vmin=0.0,
        vmax=1.0,
    )

    def process(self):
        if (
            self.weights is None
            or self.activations is None
            or self.input_kernel_size is None
        ):
            return

        w = np.asarray(self.weights, dtype=np.float64)
        act = np.asarray(self.activations, dtype=np.float64)
        P = int(self.input_kernel_size)

        k, dim = w.shape[0], w.shape[1]
        C = dim // (P * P)

        if act.ndim != 3 or act.shape[2] != k:
            raise ValueError(
                f"Expected 3D activations (grid_h, grid_w, {k}), got shape {act.shape}"
            )

        grid_h, grid_w = act.shape[0], act.shape[1]
        n_patches = grid_h * grid_w

        # Reshape activations to (n_patches, K)
        act_patches = act.reshape(n_patches, k)

        # Reconstruct each patch: (n_patches, dim) = (n_patches, K) @ (K, dim)
        recon_patches = act_patches @ w

        # Add back per-patch means (only for first layer with ConvLGN)
        if self.input_means is not None:
            means = np.asarray(self.input_means, dtype=np.float64)
            if means.shape[0] != n_patches:
                raise ValueError(
                    f"patch means length {means.shape[0]} does not match patch count {n_patches}"
                )
            recon_patches = recon_patches + means[:, None]

        # Assemble patches back into spatial array
        if C == 1:
            # 2D output: (grid_h*P, grid_w*P)
            result = (
                recon_patches.reshape(grid_h, grid_w, P, P)
                .transpose(0, 2, 1, 3)
                .reshape(grid_h * P, grid_w * P)
            )
        else:
            # 3D output: (grid_h*P, grid_w*P, C)
            result = (
                recon_patches.reshape(grid_h, grid_w, P, P, C)
                .transpose(0, 2, 1, 3, 4)
                .reshape(grid_h * P, grid_w * P, C)
            )

        self.result = result
        self.result_heatmap = to_display_grid(result if result.ndim == 2 else result.mean(axis=2))
