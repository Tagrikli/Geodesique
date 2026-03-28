"""TopK — keeps only the top-k activations by absolute magnitude."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Integer
from axonforge.core.descriptors.displays import Text


@branch("Hypercolumn")
class TopK(Node):
    activations = InputPort("Activations", np.ndarray)

    result = OutputPort("Result", np.ndarray)

    k = Integer("K", default=1)
    info = Text("Info", default="idle")

    def process(self):
        if self.activations is None:
            return

        a = np.asarray(self.activations, dtype=np.float64)
        k = min(int(self.k), a.size)

        top_indices = np.argpartition(np.abs(a), -k)[-k:]
        out = np.zeros_like(a)
        out[top_indices] = a[top_indices]

        self.result = out
        self.info = f"top-{k} of {a.size}  max={a[top_indices[np.argmax(np.abs(a[top_indices]))]]:+.3f}"
