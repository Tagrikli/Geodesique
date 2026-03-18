import math
import numpy as np
from typing import Optional

from axonforge.core.node import Node
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range, Integer, Bool
from axonforge.core.descriptors.displays import Heatmap
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors import branch
from .utilities import to_display_grid


class Weights(Node):
    """Generate and hold a weight matrix of unit vectors."""

    amount = Integer("Amount", default=25)
    dim = Integer("Dim", default=784)

    update = InputPort("Update", np.ndarray)
    weights = OutputPort("Weights", np.ndarray)
    amount_out = OutputPort("Amount", int)

    preview = Heatmap("Weights", colormap="bwr")

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        self._on_reset()
        self.preview = to_display_grid(self._weights)

    def _on_reset(self, params=None):
        cols = int(self.dim)
        rows = int(self.amount)
        w = np.random.randn(rows, cols)
        norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self._weights = w / norms
        self.preview = to_display_grid(self._weights)
        self._ignore_update_once = True
        return {"status": "ok"}

    def process(self):
        if getattr(self, "_ignore_update_once", False):
            self._ignore_update_once = False
        elif self.update is not None:
            w = self.update
            norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
            self._weights = w / norms
        self.weights = self._weights
        self.amount_out = self.amount
        self.preview = to_display_grid(self._weights)
