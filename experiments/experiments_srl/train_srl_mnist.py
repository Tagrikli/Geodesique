"""Train SignedResidualLearning dictionary on MNIST — continuous online training.

Standalone experiment script mirroring the SRL node (learning4.py): fully signed
activations (no ReLU), signed reconstruction, sign-aware LOO residuals, and
geodesic (Rodrigues) weight updates for all templates.

Tracks Participation Ratio (energy) and Gini coefficient (|a_inh|) as density
metrics over a single continuous training run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm

try:
    import cupy as xp
    USING_CUPY = True
except Exception:
    import numpy as xp  # type: ignore[no-redef]
    USING_CUPY = False


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "ax_data" / "datasets"

# Dataset selection: "digits" or "fashion"
DATASET = "fashion"
OUTPUT_DIR = ROOT / "experiments_srl" / f"r007_{DATASET}"

# Run
SEED = 42
TRAIN_SAMPLES = 30_000 * 3 # Total presentations (loops over dataset if needed)

# SRL hyperparameters
TEMPLATE_COUNT = 1024
INPUT_DIM = 28 * 28
LEARNING_RATE = 0.1
EPS = 1e-8

# Logging
LOG_EVERY = 500  # Compute running averages over this window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(a):
    if USING_CUPY:
        return xp.asnumpy(a)
    return np.asarray(a)


def _participation_ratio(a) -> float:
    """PR = (sum p_i)^2 / sum(p_i^2) where p_i = a_i^2 (energy)."""
    energy = a ** 2
    total = float(_to_numpy(xp.sum(energy)))
    if total <= EPS:
        return 0.0
    return total ** 2 / float(_to_numpy(xp.sum(energy ** 2)))


def _gini_coefficient(a) -> float:
    """Gini on |a_i|."""
    vals = xp.sort(xp.abs(a))
    n = vals.shape[0]
    if n <= 1:
        return 0.0
    total = float(_to_numpy(xp.sum(vals)))
    if total <= EPS:
        return 0.0
    index = xp.arange(1, n + 1, dtype=vals.dtype)
    numer = float(_to_numpy(xp.sum((2.0 * index - n - 1.0) * vals)))
    return numer / (n * total)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_mnist_train() -> np.ndarray:
    """Load full MNIST training set from pre-saved npy files."""
    prefix = "fashion_mnist" if DATASET == "fashion" else "mnist"
    path = DATA_DIR / f"{prefix}_train_images.npy"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    images = np.load(path)
    images = images.reshape(images.shape[0], -1)  # flatten to (N, 784)
    return images.astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# SRL online step (mirrors learning4.py)
# ---------------------------------------------------------------------------

def _srl_online_step(x, w, eta: float):
    """One SRL sample step. Returns (w_new, metrics_tuple) or (w, None)."""
    eps = EPS

    # 1. Mean-center and L2-normalize
    x_c = x - xp.mean(x)
    x_c_norm = xp.linalg.norm(x_c)
    if x_c_norm < eps:
        return w, None
    x_hat = x_c / (x_c_norm + eps)

    # Renormalize templates
    w = w / (xp.linalg.norm(w, axis=1, keepdims=True) + eps)

    # 2. Signed activations (no ReLU)
    s = w @ x_hat  # (k,)

    # 3. Full signed reconstruction
    x_recon = s @ w  # (D,)

    # 4. LOO shadow & sign-aware residuals
    x_shadow = x_recon[None, :] - (s[:, None] * w)  # (k, D)
    x_shadow_norm = xp.linalg.norm(x_shadow, axis=1, keepdims=True)
    x_shadow_hat = x_shadow / (x_shadow_norm + eps)

    sign_s = xp.sign(s)
    r_loos = sign_s[:, None] * (x_hat[None, :] - x_shadow_hat)  # (k, D)

    # 5. Learning (all templates, no active gate)
    r_dot_w = xp.sum(r_loos * w, axis=1, keepdims=True)
    tau = r_loos - r_dot_w * w

    tau_norm = xp.linalg.norm(tau, axis=1, keepdims=True)
    tau_hat = xp.where(tau_norm > eps, tau / tau_norm, 0.0)

    theta = eta * tau_norm
    w_new = w * xp.cos(theta) + tau_hat * xp.sin(theta)
    w_new = w_new / (xp.linalg.norm(w_new, axis=1, keepdims=True) + eps)

    # 6. LOO-inhibited activations (signed)
    shadow_overlap = xp.sum(w * x_shadow, axis=1)
    a_inh = s - shadow_overlap

    # Reconstruction MSE (stays on GPU as scalar)
    dim = w.shape[1]
    recon = x_recon * x_c_norm
    mse = xp.sum((recon - x) ** 2) / dim

    return w_new, (a_inh, mse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_data_cpu = _load_mnist_train()
    n_data = x_data_cpu.shape[0]

    if USING_CUPY:
        x_data = xp.asarray(x_data_cpu)
    else:
        x_data = x_data_cpu

    np.random.seed(SEED)
    w = np.random.randn(TEMPLATE_COUNT, INPUT_DIM).astype(np.float64)
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + EPS)
    w = xp.asarray(w)

    # Shuffled index stream that wraps around the dataset
    order = np.random.permutation(n_data)

    print(json.dumps({
        "algo": "SignedResidualLearning",
        "device": "cupy-gpu" if USING_CUPY else "numpy-cpu",
        "dataset": DATASET,
        "dataset_size": n_data,
        "train_samples": TRAIN_SAMPLES,
        "template_count": TEMPLATE_COUNT,
        "learning_rate": LEARNING_RATE,
    }, indent=2))

    history: list[dict[str, float]] = []
    all_pr: list[float] = []
    all_gini: list[float] = []

    # Buffers on GPU, compute averages at LOG_EVERY
    a_inh_buf = xp.zeros((LOG_EVERY, TEMPLATE_COUNT), dtype=np.float64)
    mse_buf = xp.zeros(LOG_EVERY, dtype=np.float64)
    buf_idx = 0

    progress = tqdm(range(TRAIN_SAMPLES), desc="training", leave=True, mininterval=0.5)

    for step in progress:
        idx = order[step % n_data]
        # Reshuffle when we wrap around
        if step > 0 and step % n_data == 0:
            order = np.random.permutation(n_data)

        w, result = _srl_online_step(x_data[idx], w, LEARNING_RATE)

        if result is None:
            continue

        a_inh, mse = result
        a_inh_buf[buf_idx] = a_inh
        mse_buf[buf_idx] = mse
        buf_idx += 1

        if (step + 1) % LOG_EVERY == 0 and buf_idx > 0:
            # Compute PR and Gini per sample in buffer, then average
            buf = a_inh_buf[:buf_idx]
            prs = [_participation_ratio(buf[i]) for i in range(buf_idx)]
            ginis = [_gini_coefficient(buf[i]) for i in range(buf_idx)]
            mean_pr = sum(prs) / len(prs)
            mean_gini = sum(ginis) / len(ginis)
            mean_mse = float(_to_numpy(xp.mean(mse_buf[:buf_idx])))

            all_pr.extend(prs)
            all_gini.extend(ginis)

            record = {
                "step": float(step + 1),
                "mse": mean_mse,
                "pr": mean_pr,
                "gini": mean_gini,
            }
            history.append(record)

            progress.set_postfix(
                mse=f"{mean_mse:.5f}",
                PR=f"{mean_pr:.2f}",
                Gini=f"{mean_gini:.4f}",
            )
            buf_idx = 0

    # Save
    w_cpu = _to_numpy(w)
    np.savez(
        OUTPUT_DIR / "srl_mnist_latest.npz",
        weights=w_cpu,
        pr_values=np.array(all_pr),
        gini_values=np.array(all_gini),
    )
    with (OUTPUT_DIR / "srl_mnist_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    visualize(history, w_cpu, np.array(all_pr), np.array(all_gini))


def visualize(history: list[dict], weights: np.ndarray,
              all_pr: np.ndarray, all_gini: np.ndarray):
    """Generate training diagnostics figure."""
    steps = [r["step"] for r in history]

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    fig.suptitle("SignedResidualLearning — MNIST Training Diagnostics", fontsize=14, y=1.01)
    gs = GridSpec(3, 4, figure=fig)

    # Row 1: training curves

    # 1. Reconstruction MSE
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, [r["mse"] for r in history], "-", color="tab:red", lw=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE")

    # 2. Participation Ratio
    ax = fig.add_subplot(gs[0, 1])
    pr_vals = [r["pr"] for r in history]
    ax.plot(steps, pr_vals, "-", color="tab:blue", lw=1)
    ax.margins(y=0.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("PR")
    ax.set_title("Participation Ratio (energy)")

    # 3. Gini Coefficient
    ax = fig.add_subplot(gs[0, 2])
    gini_vals = [r["gini"] for r in history]
    ax.plot(steps, gini_vals, "-", color="tab:orange", lw=1)
    ax.margins(y=0.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gini")
    ax.set_title("Gini Coefficient (|a_inh|)")

    # 4. PR & Gini overlay
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(steps, [r["pr"] for r in history], "-", color="tab:blue", lw=1, label="PR")
    ax.set_xlabel("Step")
    ax.set_ylabel("PR", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(steps, [r["gini"] for r in history], "-", color="tab:orange", lw=1, label="Gini")
    ax2.set_ylabel("Gini", color="tab:orange")
    ax.set_title("PR & Gini (overlay)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # Row 2: weight grid + PR distribution

    n_templates = weights.shape[0]
    grid_cols = int(math.ceil(math.sqrt(n_templates)))
    grid_rows = int(math.ceil(n_templates / grid_cols))
    img_h, img_w = 28, 28

    canvas = np.zeros((grid_rows * img_h, grid_cols * img_w))
    for idx in range(n_templates):
        r, c = divmod(idx, grid_cols)
        tile = weights[idx].reshape(img_h, img_w)
        canvas[r * img_h:(r + 1) * img_h, c * img_w:(c + 1) * img_w] = tile

    ax = fig.add_subplot(gs[1, :3])
    ax.imshow(canvas, cmap="bwr", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"Learned Templates ({n_templates}) — raw weights [-1, 1]")
    ax.axis("off")

    # PR histogram (all samples)
    ax = fig.add_subplot(gs[1, 3])
    if all_pr.size > 0:
        ax.hist(all_pr, bins=50, color="tab:blue", edgecolor="black", alpha=0.7)
        ax.axvline(all_pr.mean(), color="red", ls="--", label=f"mean={all_pr.mean():.1f}")
        ax.legend(fontsize=7)
    ax.set_xlabel("PR")
    ax.set_ylabel("Count")
    ax.set_title("PR Distribution")

    # Row 3: Gini histogram + scatter

    ax = fig.add_subplot(gs[2, 0])
    if all_gini.size > 0:
        ax.hist(all_gini, bins=50, color="tab:orange", edgecolor="black", alpha=0.7)
        ax.axvline(all_gini.mean(), color="red", ls="--", label=f"mean={all_gini.mean():.3f}")
        ax.legend(fontsize=7)
    ax.set_xlabel("Gini")
    ax.set_ylabel("Count")
    ax.set_title("Gini Distribution")

    ax = fig.add_subplot(gs[2, 1])
    if all_pr.size > 0 and all_gini.size > 0:
        ax.scatter(all_gini, all_pr, s=2, alpha=0.3, color="tab:purple")
    ax.set_xlabel("Gini")
    ax.set_ylabel("PR")
    ax.set_title("PR vs Gini")

    fig_path = OUTPUT_DIR / "srl_mnist_diagnostics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostics figure to {fig_path}")


if __name__ == "__main__":
    main()
