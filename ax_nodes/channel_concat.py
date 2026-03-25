"""ChannelConcat — mean-centers and L2-normalizes each channel, then vstacks."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Range
from axonforge.core.descriptors.displays import Text, Heatmap


@branch("Hypercolumn")
class ChannelConcat(Node):
    """Independently mean-centers and L2-normalizes up to 4 input channels,
    then vertically stacks them so each channel has equal influence under L2 norm.

    All active channels must share the same column size (i.e. each channel's
    flattened length must be a multiple of a common column width).  The output
    is a 2D array where each channel contributes one or more rows of that width.
    """

    ch0 = InputPort("Channel 0", np.ndarray)
    ch1 = InputPort("Channel 1", np.ndarray)
    ch2 = InputPort("Channel 2", np.ndarray)
    ch3 = InputPort("Channel 3", np.ndarray)

    w0 = Range("Weight 0", default=1.0, min_val=0.0, max_val=1.0, step=0.01)
    w1 = Range("Weight 1", default=1.0, min_val=0.0, max_val=1.0, step=0.01)
    w2 = Range("Weight 2", default=1.0, min_val=0.0, max_val=1.0, step=0.01)
    w3 = Range("Weight 3", default=1.0, min_val=0.0, max_val=1.0, step=0.01)

    output = OutputPort("Output", np.ndarray)
    channel_sizes = OutputPort("Channel Sizes", np.ndarray)

    preview = Heatmap("Output", colormap="bwr", vmin=-1, vmax=1, scale_mode="manual")
    info = Text("Info", default="idle")

    def process(self):
        eps = 1e-8
        channels = [self.ch0, self.ch1, self.ch2, self.ch3]
        weights = [float(self.w0), float(self.w1), float(self.w2), float(self.w3)]

        rows = []
        sizes = []
        active_indices = []
        col_size = None

        for i, ch in enumerate(channels):
            if ch is None:
                continue

            x = np.asarray(ch, dtype=np.float64)

            # Determine column size from the first active channel
            if x.ndim == 2:
                cols = x.shape[1]
            elif x.ndim == 1:
                cols = x.size
            else:
                cols = x.shape[-1]

            if col_size is None:
                col_size = cols
            elif cols != col_size:
                self.info = f"ch{i} col size {cols} != {col_size}"
                return

            x = x.ravel()
            if x.size % col_size != 0:
                self.info = f"ch{i} size {x.size} not divisible by col_size {col_size}"
                return

            # Mean-center
            x = x - np.mean(x)

            # L2-normalize (pass through as-is if zero-norm), then scale
            norm = np.linalg.norm(x)
            if norm >= eps:
                x = x / norm
            x = x * weights[i]
            rows.append(x.reshape(-1, col_size))
            sizes.append(x.size)
            active_indices.append(i)

        if len(rows) == 0:
            self.info = "No active channels"
            return

        stacked = np.vstack(rows)
        self.output = stacked
        self.channel_sizes = np.array(sizes, dtype=np.int64)
        self.preview = stacked
        self.info = "  ".join(f"ch{i}:{s}" for i, s in zip(active_indices, sizes))
