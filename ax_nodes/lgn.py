"""LGN — mean-centers the input, preserving shape."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Integer
from axonforge.core.descriptors.displays import Numeric, Heatmap


@branch("Hypercolumn")
class LGN(Node):
    """Mean-centers the input and outputs the result with its original shape."""

    input_data = InputPort("Input", np.ndarray)
    output_data = OutputPort("Output", np.ndarray)
    output_mean = OutputPort("Mean", float)

    mean_value = Numeric("Mean")

    def process(self):
        if self.input_data is None:
            return

        mean = np.mean(self.input_data)
        self.output_data = self.input_data - mean
        self.output_mean = mean
        self.mean_value = mean


@branch("Hypercolumn/Conv")
class ConvConfig(Node):
    """Configuration node for convolutional pipeline. Sets kernel size (stride = kernel size)."""

    kernel_size = Integer("Kernel Size", default=4)

    output_kernel_size = OutputPort("Kernel Size", int)

    def process(self):
        self.output_kernel_size = int(self.kernel_size)


@branch("Hypercolumn/Conv")
class ConvLGN(Node):
    """Convolutional LGN: mean-centers each non-overlapping patch in-place.

    Takes a 2D image and kernel_size from ConvConfig.  Tiles the image into
    kernel_size × kernel_size non-overlapping patches, mean-centers each patch,
    and outputs the full 2D image with patches mean-centered in-place.

    Input Ports:
        input_data:  2D input image (np.ndarray, H × W or flattened)
        input_kernel_size:  Kernel size from ConvConfig (int)

    Output Ports:
        output_data:   Full 2D image with per-patch mean subtracted
        output_means:  Per-patch means (n_patches,) for reconstruction
        output_kernel_size:  Kernel size pass-through (int)
    """

    input_data = InputPort("Input", np.ndarray)
    input_kernel_size = InputPort("Kernel Size", int)

    output_data = OutputPort("Output", np.ndarray)
    output_means = OutputPort("Means", np.ndarray)
    output_kernel_size = OutputPort("Kernel Size", int)

    preview = Heatmap("Output", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual")

    def process(self):
        if self.input_data is None or self.input_kernel_size is None:
            return

        inp = np.asarray(self.input_data, dtype=np.float64)
        P = int(self.input_kernel_size)

        # Ensure 2D
        if inp.ndim == 1:
            side = int(np.sqrt(inp.size))
            inp = inp.reshape(side, side)

        H, W = inp.shape
        if H % P != 0 or W % P != 0:
            raise ValueError(
                f"ConvLGN input shape {inp.shape} is not divisible by kernel size {P}"
            )
        grid_h = H // P
        grid_w = W // P

        # Tile into patches, mean-center, then put back
        patches = (
            inp.reshape(grid_h, P, grid_w, P)
            .transpose(0, 2, 1, 3)
            .reshape(grid_h * grid_w, P * P)
        )

        means = np.mean(patches, axis=1)
        patches = patches - means[:, None]

        # Reassemble into full 2D image
        result = (
            patches.reshape(grid_h, grid_w, P, P)
            .transpose(0, 2, 1, 3)
            .reshape(grid_h * P, grid_w * P)
        )

        self.output_data = result
        self.output_means = means
        self.output_kernel_size = P
        self.preview = result
