"""RandomUnitVector — emits a random mean-centered unit vector each tick."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import OutputPort
from axonforge.core.descriptors.fields import Integer


@branch("Utilities")
class RandomUnitVector(Node):
    output = OutputPort("Output", np.ndarray)

    length = Integer("Length", default=28)

    def process(self):
        n = int(self.length)
        v = np.random.randn(n)
        v -= np.mean(v)
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v /= norm
        self.output = v
