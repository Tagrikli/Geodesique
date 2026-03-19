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
