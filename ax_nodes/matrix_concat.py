"""MatrixConcat — vertically stacks two same-sized 2D arrays."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.displays import Heatmap


@branch("Utility")
class MatrixConcat(Node):
    """Vertically concatenates two 2D arrays with matching column count."""

    input_a = InputPort("A", np.ndarray)
    input_b = InputPort("B", np.ndarray)

    output_result = OutputPort("Result", np.ndarray)

    preview = Heatmap("Result")

    def process(self):
        if self.input_a is None or self.input_b is None:
            return

        a = np.asarray(self.input_a)
        b = np.asarray(self.input_b)

        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError(
                f"MatrixConcat requires 2D arrays with matching columns, got {a.shape} and {b.shape}"
            )

        result = np.vstack((a, b))
        self.output_result = result
        self.preview = result
