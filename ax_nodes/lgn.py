"""LGN — mean-centers the input, preserving shape."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.displays import Numeric


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
