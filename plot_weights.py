import json
import numpy as np
import matplotlib.pyplot as plt
from ax_nodes.utilities import to_display_grid

with open("ax_graphs/tutorial3.json") as f:
    graph = json.load(f)

node = next(n for n in graph["nodes"] if n["type"] == "SignedResidualLearning")
npy_path = node["states"]["w"]["file"]

w = np.load(npy_path)
grid = to_display_grid(w)
vmax = max(abs(grid.min()), abs(grid.max()))

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(grid, cmap="bwr", vmin=-vmax, vmax=vmax)
ax.set_title("SignedResidualLearning weights")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("weights_heatmap.png", dpi=150)
print(f"Saved weights_heatmap.png  (raw={w.shape}, grid={grid.shape})")
