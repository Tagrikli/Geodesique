import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.displays import Text


@branch("Color/Opponent")
class ColorOpponent(Node):
    image = InputPort("Image", np.ndarray)

    opponent = OutputPort("Opponent (L, RG, BG)", np.ndarray)
    luminance = OutputPort("L", np.ndarray)
    rg = OutputPort("RG", np.ndarray)
    bg = OutputPort("BG", np.ndarray)

    info = Text("Info", default="idle")

    def process(self):
        if self.image is None:
            self.info = "No input"
            return

        img = np.asarray(self.image, dtype=np.float64)

        if img.ndim != 3 or img.shape[2] not in (3, 4):
            self.info = f"Expected HxWx3 or HxWx4, got {img.shape}"
            return

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        l_ch = 0.299 * r + 0.587 * g + 0.114 * b
        rg_ch = r - g
        bg_ch = b - g

        self.luminance = l_ch
        self.rg = rg_ch
        self.bg = bg_ch
        self.opponent = np.stack([l_ch, rg_ch, bg_ch], axis=-1)
        self.info = f"{img.shape[:2]} -> 3ch opponent"
