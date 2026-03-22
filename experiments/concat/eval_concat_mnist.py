"""Train concat-SRL on MNIST and evaluate digit classification via cosine similarity.

Pipeline (from concat.json graph):
  Training:
    1. MNIST image (28x28) -> LGN (mean-center) -> top block (28x28)
    2. Digit label -> SparseIndexHashEncode (28-dim binary) -> reshape (1,28)
       -> LGN (mean-center) -> bottom row (1x28)
    3. MatrixConcat: vstack(top, bottom) -> (29, 28) -> flatten (812) -> SRL (online learning)

  Inference:
    1. Same image pipeline but bottom row = zeros (1x28)
    2. Reconstruct from frozen SRL weights+activations, extract bottom row
    3. Cosine similarity of predicted bottom row vs each digit's mean-centered hash code
    4. Argmax = predicted class
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from mnist import MNIST
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-8
IMG_ROWS, IMG_COLS = 28, 28         # MNIST image dimensions

# ---- Configuration -------------------------------------------------------
DATASET = "digits"                  # "digits" or "fashion"
SEED = 42
TRAIN_SAMPLES = 60_000 * 3              # Total training presentations (stratified)
TEMPLATE_COUNT = 500                # Number of SRL templates (from graph)
LEARNING_RATE = 0.01               # SRL step fraction η (from graph)
SPARSE_NUM_ONES = 1               # Number of 1s in each hash vector
SPARSE_VECTOR_SIZE = 28            # Hash vector length (must be multiple of IMG_COLS)
LOG_EVERY = 500                    # Training log interval
OUTPUT_DIR = ROOT / "experiments" / "concat" / "r002_digits"
# ---------------------------------------------------------------------------

# Derived constants
HASH_ROWS = SPARSE_VECTOR_SIZE // IMG_COLS   # rows the hash occupies after reshape
TOTAL_ROWS = IMG_ROWS + HASH_ROWS            # total rows of concat input
INPUT_DIM = TOTAL_ROWS * IMG_COLS             # flattened input size for SRL


# ---------------------------------------------------------------------------
# Sparse hash encoding (replicates SparseIndexHashEncode node)
# ---------------------------------------------------------------------------

def sparse_hash_encode(value: int, num_ones: int = 10, size: int = 28) -> np.ndarray:
    """Deterministic hash of integer to sparse binary vector.

    Uses SHA-256 for cross-session determinism (Python's built-in hash()
    is randomized per process in Python 3.3+).
    """
    num_ones = max(0, min(size, num_ones))
    hash_val = int(hashlib.sha256(str(value).encode()).hexdigest(), 16)
    rng = np.random.RandomState(hash_val % (2**31))
    output = np.zeros(size, dtype=np.float64)
    if num_ones > 0:
        indices = rng.choice(size, size=num_ones, replace=False)
        output[indices] = 1.0
    return output


def build_digit_codebook(n_classes: int = 10) -> np.ndarray:
    """Pre-compute sparse hash codes for digits 0..n_classes-1.

    Guarantees collision-free codes when C(vector_size, num_ones) >= n_classes.
    Raises ValueError if uniqueness is impossible.

    Returns (n_classes, SPARSE_VECTOR_SIZE) array of raw (un-mean-centered) codes.
    """
    max_unique = math.comb(SPARSE_VECTOR_SIZE, SPARSE_NUM_ONES)
    if max_unique < n_classes:
        raise ValueError(
            f"Cannot create {n_classes} unique codes with {SPARSE_NUM_ONES} ones "
            f"in {SPARSE_VECTOR_SIZE}-dim vectors: only C({SPARSE_VECTOR_SIZE}, "
            f"{SPARSE_NUM_ONES}) = {max_unique} unique patterns exist"
        )

    codes = np.zeros((n_classes, SPARSE_VECTOR_SIZE), dtype=np.float64)
    seen: set[tuple[float, ...]] = set()
    for d in range(n_classes):
        salt = 0
        while True:
            candidate = sparse_hash_encode(
                d + salt * n_classes, SPARSE_NUM_ONES, SPARSE_VECTOR_SIZE,
            )
            key = tuple(candidate.tolist())
            if key not in seen:
                seen.add(key)
                codes[d] = candidate
                break
            salt += 1
    return codes


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_mnist_path(dataset: str = "digits") -> Path:
    if dataset == "fashion":
        candidates = [
            ROOT / "ax_data" / "datasets" / "raw" / "fashion-mnist",
            ROOT / "data" / "mnist" / "fashion-mnist",
        ]
    else:
        candidates = [
            ROOT / "ax_data" / "datasets" / "raw" / "mnist",
            ROOT / "data" / "mnist" / "mnist",
            ROOT / "data" / "mnist_data",
        ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"MNIST ({dataset}) dataset not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _load_mnist(dataset: str = "digits"):
    mnist_path = _resolve_mnist_path(dataset)
    loader = MNIST(str(mnist_path))

    x_train, y_train = loader.load_training()
    x_train = np.asarray(x_train, dtype=np.float64) / 255.0
    y_train = np.asarray(y_train)

    x_test, y_test = loader.load_testing()
    x_test = np.asarray(x_test, dtype=np.float64) / 255.0
    y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Input construction (replicates graph pipeline)
# ---------------------------------------------------------------------------

def _build_input(image_flat: np.ndarray, hash_vec: np.ndarray) -> tuple[np.ndarray, float]:
    """Build the concatenated input matching the graph pipeline.

    1. image (784,) -> reshape (28,28) -> LGN (mean-center)
    2. hash_vec (SPARSE_VECTOR_SIZE,) -> reshape (HASH_ROWS,28) -> LGN (mean-center)
    3. vstack -> (TOTAL_ROWS, 28) -> ravel -> (INPUT_DIM,)

    Returns (input_flat, image_mean).
    """
    img = image_flat.reshape(IMG_ROWS, IMG_COLS)
    img_mean = float(np.mean(img))
    img_lgn = img - img_mean

    hash_2d = hash_vec.reshape(HASH_ROWS, IMG_COLS)
    hash_mean = float(np.mean(hash_2d))
    hash_lgn = hash_2d - hash_mean

    concat = np.vstack([img_lgn, hash_lgn])  # (TOTAL_ROWS, IMG_COLS)
    return concat.ravel(), img_mean


# ---------------------------------------------------------------------------
# SRL step (mirrors learning4.py / train_srl_mnist.py)
# ---------------------------------------------------------------------------

def _srl_step(x: np.ndarray, w: np.ndarray, eta: float, learn: bool = True):
    """Single SRL online step.

    Returns (w_new, a_inh_scaled, mse).
    If input norm is near zero, returns (w, None, None).
    """
    x_norm = np.linalg.norm(x)
    if x_norm < EPS:
        return w, None, None
    x_hat = x / x_norm

    # Renormalize templates
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + EPS)

    # Signed activations
    s = w @ x_hat  # (K,)

    # Full signed reconstruction
    x_recon = s @ w  # (D,)

    # LOO shadow
    x_shadow = x_recon[None, :] - (s[:, None] * w)  # (K, D)
    x_shadow_norm = np.linalg.norm(x_shadow, axis=1, keepdims=True)
    x_shadow_hat = x_shadow / (x_shadow_norm + EPS)

    # Sign-aware residuals
    sign_s = np.sign(s)
    r_loos = sign_s[:, None] * (x_hat[None, :] - x_shadow_hat)

    if learn:
        r_dot_w = np.sum(r_loos * w, axis=1, keepdims=True)
        tau = r_loos - r_dot_w * w
        tau_norm = np.linalg.norm(tau, axis=1, keepdims=True)
        tau_hat = np.where(tau_norm > EPS, tau / tau_norm, 0.0)
        theta = eta * tau_norm
        w_new = w * np.cos(theta) + tau_hat * np.sin(theta)
        w_new = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + EPS)
    else:
        w_new = w

    # LOO-inhibited activations (scaled by input norm)
    shadow_overlap = np.sum(w * x_shadow, axis=1)
    a_inh = (s - shadow_overlap) * x_norm

    # Reconstruction MSE
    recon_scaled = x_recon * x_norm
    mse = float(np.sum((recon_scaled - x) ** 2) / x.shape[0])

    return w_new, a_inh, mse


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _build_stratified_order(
    y: np.ndarray, n_samples: int, rng: np.random.RandomState,
) -> np.ndarray:
    """Build a stratified sample order balanced across classes."""
    classes = np.unique(y)
    n_classes = len(classes)
    indices_by_class = {int(c): np.where(y == c)[0] for c in classes}

    for c in classes:
        rng.shuffle(indices_by_class[int(c)])

    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    order = []
    for i, c in enumerate(classes):
        n_c = samples_per_class + (1 if i < remainder else 0)
        class_idx = indices_by_class[int(c)]
        if n_c > len(class_idx):
            repeats = (n_c // len(class_idx)) + 1
            class_idx = np.tile(class_idx, repeats)
        order.append(class_idx[:n_c])

    order = np.concatenate(order)
    rng.shuffle(order)
    return order


# ---------------------------------------------------------------------------
# Training-time hash metrics
# ---------------------------------------------------------------------------

def _extract_hash_from_recon(recon_flat: np.ndarray) -> np.ndarray:
    """Extract the hash portion (bottom rows) from a flattened reconstruction."""
    return recon_flat.reshape(TOTAL_ROWS, IMG_COLS)[IMG_ROWS:, :].ravel()


def _hash_dot(recon_flat: np.ndarray, target_hash_lgn: np.ndarray) -> float:
    """Dot product between reconstructed hash rows and target hash."""
    pred = _extract_hash_from_recon(recon_flat)
    return float(np.dot(pred, target_hash_lgn))


def _hash_mse(recon_flat: np.ndarray, target_hash_lgn: np.ndarray) -> float:
    """MSE between reconstructed hash rows and target hash."""
    pred = _extract_hash_from_recon(recon_flat)
    return float(np.mean((pred - target_hash_lgn) ** 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_and_evaluate():
    if SPARSE_VECTOR_SIZE % IMG_COLS != 0:
        raise ValueError(
            f"SPARSE_VECTOR_SIZE ({SPARSE_VECTOR_SIZE}) must be a multiple of "
            f"IMG_COLS ({IMG_COLS}) so the hash vector can be reshaped to (-1, {IMG_COLS})"
        )

    rng = np.random.RandomState(SEED)

    # Load data
    print(f"Loading MNIST ({DATASET})...")
    x_train, y_train, x_test, y_test = _load_mnist(DATASET)
    print(f"  Train: {x_train.shape[0]}  Test: {x_test.shape[0]}")

    # Build digit codebook
    n_classes = len(np.unique(y_train))
    codebook = build_digit_codebook(n_classes)
    codebook_lgn = codebook - codebook.mean(axis=1, keepdims=True)

    print(f"\nDigit codebook ({SPARSE_VECTOR_SIZE}-dim, {SPARSE_NUM_ONES} ones):")
    for d in range(n_classes):
        print(f"  Digit {d}: {int(codebook[d].sum())} ones")

    # Check pairwise cosine similarity of codebook
    code_norms = codebook_lgn / (np.linalg.norm(codebook_lgn, axis=1, keepdims=True) + EPS)
    codebook_sim = code_norms @ code_norms.T
    off_diag = codebook_sim[~np.eye(n_classes, dtype=bool)]
    print(f"  Codebook pairwise cos sim: mean={off_diag.mean():.3f}, "
          f"max={off_diag.max():.3f}, min={off_diag.min():.3f}")

    # Initialize SRL weights
    w = rng.randn(TEMPLATE_COUNT, INPUT_DIM)
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + EPS)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    print(f"\nTraining: {TRAIN_SAMPLES} samples, {TEMPLATE_COUNT} templates, "
          f"η={LEARNING_RATE}")

    order = _build_stratified_order(y_train, TRAIN_SAMPLES, rng)

    history: list[dict[str, float]] = []
    mse_buf: list[float] = []
    hash_cos_buf: list[float] = []
    hash_mse_buf: list[float] = []

    progress = tqdm(range(TRAIN_SAMPLES), desc="Training", leave=True, mininterval=0.5)
    for step in progress:
        idx = order[step]
        digit = y_train[idx]
        x_input, _ = _build_input(x_train[idx], codebook[digit])

        w, a_inh, mse = _srl_step(x_input, w, LEARNING_RATE, learn=True)

        if mse is not None and a_inh is not None:
            mse_buf.append(mse)
            # Reconstruct and measure hash-specific metrics
            recon = w.T @ a_inh
            hash_target = codebook_lgn[digit]
            hash_cos_buf.append(_hash_dot(recon, hash_target))
            hash_mse_buf.append(_hash_mse(recon, hash_target))

        if (step + 1) % LOG_EVERY == 0 and len(mse_buf) > 0:
            record = {
                "step": float(step + 1),
                "mse": float(np.mean(mse_buf)),
                "hash_cos": float(np.mean(hash_cos_buf)),
                "hash_mse": float(np.mean(hash_mse_buf)),
            }
            history.append(record)
            progress.set_postfix(
                mse=f"{record['mse']:.5f}",
                h_cos=f"{record['hash_cos']:.3f}",
                h_mse=f"{record['hash_mse']:.5f}",
            )
            mse_buf.clear()
            hash_cos_buf.clear()
            hash_mse_buf.clear()

    if history:
        print(f"Training complete. Final MSE={history[-1]['mse']:.6f}, "
              f"Hash cos={history[-1]['hash_cos']:.3f}, "
              f"Hash MSE={history[-1]['hash_mse']:.6f}")

    # -----------------------------------------------------------------------
    # Inference on test set
    # -----------------------------------------------------------------------
    print(f"\nEvaluating on {x_test.shape[0]} test samples...")

    # Renormalize weights once before inference
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + EPS)

    n_test = x_test.shape[0]
    predicted_rows = np.zeros((n_test, SPARSE_VECTOR_SIZE), dtype=np.float64)
    zeros_row = np.zeros(SPARSE_VECTOR_SIZE)

    for i in tqdm(range(n_test), desc="Inference", leave=False):
        x_input, _img_mean = _build_input(x_test[i], zeros_row)

        _, a_inh, _ = _srl_step(x_input, w, 0.0, learn=False)

        if a_inh is None:
            continue

        # Reconstruct (W.T @ activations), no mean added — extract hash rows
        recon = w.T @ a_inh  # (INPUT_DIM,) in LGN space
        predicted_rows[i] = _extract_hash_from_recon(recon)

    # -----------------------------------------------------------------------
    # Classification via dot product
    # -----------------------------------------------------------------------
    # Compare predicted rows with mean-centered codebook
    all_sims = predicted_rows @ codebook_lgn.T  # (n_test, n_classes)

    y_pred = np.argmax(all_sims, axis=1)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    top3_acc = top_k_accuracy_score(y_test, all_sims, k=3, labels=np.arange(n_classes))
    top5_acc = top_k_accuracy_score(y_test, all_sims, k=5, labels=np.arange(n_classes))
    report_str = str(classification_report(y_test, y_pred, digits=4))
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Test accuracy:  {acc:.4f}")
    print(f"  Top-3 accuracy: {top3_acc:.4f}")
    print(f"  Top-5 accuracy: {top5_acc:.4f}")
    print(f"{'='*50}\n")
    print(report_str)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun directory: {output_dir}")

    np.save(output_dir / "srl_weights.npy", w)
    np.save(output_dir / "digit_codebook.npy", codebook)

    # Save classification report as text
    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Concat-SRL MNIST Classification Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset:        {DATASET}\n")
        f.write(f"Train samples:  {TRAIN_SAMPLES}\n")
        f.write(f"Templates:      {TEMPLATE_COUNT}\n")
        f.write(f"Learning rate:  {LEARNING_RATE}\n")
        f.write(f"Num ones:       {SPARSE_NUM_ONES}\n")
        f.write(f"Vector size:    {SPARSE_VECTOR_SIZE}\n")
        f.write(f"Seed:           {SEED}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Test accuracy:  {acc:.4f}\n")
        f.write(f"Top-3 accuracy: {top3_acc:.4f}\n")
        f.write(f"Top-5 accuracy: {top5_acc:.4f}\n\n")
        f.write(report_str)
    print(f"Saved classification report to {output_dir / 'classification_report.txt'}")

    results_json = {
        "dataset": DATASET,
        "seed": SEED,
        "train_samples": TRAIN_SAMPLES,
        "template_count": TEMPLATE_COUNT,
        "learning_rate": LEARNING_RATE,
        "sparse_num_ones": SPARSE_NUM_ONES,
        "sparse_vector_size": SPARSE_VECTOR_SIZE,
        "input_dim": INPUT_DIM,
        "test_accuracy": float(acc),
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "training_history": history,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results to {output_dir / 'results.json'}")

    # -----------------------------------------------------------------------
    # Visualize
    # -----------------------------------------------------------------------
    visualize(history, w, cm, report_dict, all_sims, y_test, codebook,
              codebook_lgn, output_dir)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(ax, cm, title: str):
    n = cm.shape[0]
    labels = list(range(n))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color=color)


def _plot_normalized_cm(ax, cm, title: str):
    n = cm.shape[0]
    labels = list(range(n))
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0.0)
    im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)


def visualize(history, w, cm, report_dict, all_sims, y_test, codebook,
              codebook_lgn, output_dir):
    n_classes = cm.shape[0]
    acc = accuracy_score(y_test, np.argmax(all_sims, axis=1))

    fig = plt.figure(figsize=(24, 24), constrained_layout=True)
    fig.suptitle(
        f"Concat-SRL MNIST Classification — {DATASET.upper()}  |  "
        f"K={TEMPLATE_COUNT}  η={LEARNING_RATE}  N={TRAIN_SAMPLES}  "
        f"acc={acc:.4f}",
        fontsize=14,
    )
    gs = GridSpec(5, 4, figure=fig)

    steps = [r["step"] for r in history] if history else []

    # Row 0, col 0: Training MSE
    ax = fig.add_subplot(gs[0, 0])
    if history:
        ax.plot(steps, [r["mse"] for r in history], "-", color="tab:red", lw=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Training Reconstruction MSE")

    # Row 0, col 1: Hash dot product over training
    ax = fig.add_subplot(gs[0, 1])
    if history:
        ax.plot(steps, [r["hash_cos"] for r in history], "-", color="tab:blue", lw=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Dot Product")
    ax.set_title("Hash Reconstruction Dot Product")

    # Row 0, col 2: Hash MSE over training
    ax = fig.add_subplot(gs[0, 2])
    if history:
        ax.plot(steps, [r["hash_mse"] for r in history], "-", color="tab:orange", lw=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Hash Reconstruction MSE")

    # Row 0, col 3: Codebook heatmap
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(codebook, cmap="binary", interpolation="nearest", aspect="auto")
    ax.set_xlabel("Vector dim")
    ax.set_ylabel("Digit")
    ax.set_yticks(range(n_classes))
    ax.set_title(f"Digit Codebook ({SPARSE_VECTOR_SIZE}-dim)")

    # Row 1: Confusion matrices
    ax = fig.add_subplot(gs[1, 0:2])
    _plot_confusion_matrix(ax, cm, "Confusion Matrix (counts)")

    ax = fig.add_subplot(gs[1, 2:4])
    _plot_normalized_cm(ax, cm, "Confusion Matrix (normalized)")

    # Row 2: Per-class metrics
    ax = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(n_classes)
    width = 0.35
    f1_scores = [report_dict[str(c)]["f1-score"] for c in range(n_classes)]
    prec_scores = [report_dict[str(c)]["precision"] for c in range(n_classes)]
    ax.bar(x_pos - width / 2, f1_scores, width, label="F1", color="tab:blue")
    ax.bar(x_pos + width / 2, prec_scores, width, label="Precision", color="tab:orange")
    ax.set_xticks(x_pos)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class F1 & Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    rec_scores = [report_dict[str(c)]["recall"] for c in range(n_classes)]
    ax.bar(x_pos, rec_scores, color="tab:green")
    ax.set_xticks(x_pos)
    ax.set_xlabel("Class")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall")
    ax.set_ylim(0, 1.05)

    # Row 2, col 2: Summary accuracy bar
    ax = fig.add_subplot(gs[2, 2])
    top3 = top_k_accuracy_score(
        y_test, all_sims, k=3, labels=np.arange(n_classes))
    top5 = top_k_accuracy_score(
        y_test, all_sims, k=5, labels=np.arange(n_classes))
    metrics = ["Top-1", "Top-3", "Top-5"]
    vals = [acc, top3, top5]
    bars = ax.bar(metrics, vals, color=["tab:blue", "tab:cyan", "tab:green"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy")
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", fontsize=8)

    # Row 2, col 3: Codebook pairwise similarity
    ax = fig.add_subplot(gs[2, 3])
    code_hat = codebook_lgn / (np.linalg.norm(codebook_lgn, axis=1, keepdims=True) + EPS)
    sim_matrix = code_hat @ code_hat.T
    im = ax.imshow(sim_matrix, cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_title("Codebook Pairwise Cos Sim")

    # Row 3: Weight grid
    n_templates = w.shape[0]
    grid_cols = int(math.ceil(math.sqrt(n_templates)))
    grid_rows = int(math.ceil(n_templates / grid_cols))
    img_h, img_w = TOTAL_ROWS, IMG_COLS

    canvas = np.zeros((grid_rows * img_h, grid_cols * img_w))
    for idx in range(n_templates):
        r, c = divmod(idx, grid_cols)
        tile = w[idx].reshape(img_h, img_w)
        canvas[r * img_h:(r + 1) * img_h, c * img_w:(c + 1) * img_w] = tile

    ax = fig.add_subplot(gs[3, :])
    ax.imshow(canvas, cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"Learned Templates ({n_templates}) — ({TOTAL_ROWS}×{IMG_COLS}) weights [-1, 1]")
    ax.axis("off")

    # Row 4: Dot product distributions
    correct_sims = all_sims[np.arange(len(y_test)), y_test]

    ax = fig.add_subplot(gs[4, 0])
    ax.hist(correct_sims, bins=50, color="tab:blue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Dot Product")
    ax.set_ylabel("Count")
    ax.set_title(f"Correct-Class Dot Prod (mean={correct_sims.mean():.3f})")

    ax = fig.add_subplot(gs[4, 1])
    wrong_mask = np.ones_like(all_sims, dtype=bool)
    wrong_mask[np.arange(len(y_test)), y_test] = False
    wrong_sims = all_sims.copy()
    wrong_sims[~wrong_mask] = -np.inf
    max_wrong = wrong_sims.max(axis=1)
    ax.hist(max_wrong, bins=50, color="tab:red", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Dot Product")
    ax.set_ylabel("Count")
    ax.set_title(f"Max Wrong-Class Dot Prod (mean={max_wrong.mean():.3f})")

    ax = fig.add_subplot(gs[4, 2])
    margin = correct_sims - max_wrong
    ax.hist(margin, bins=50, color="tab:purple", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Margin")
    ax.set_ylabel("Count")
    ax.set_title(f"Margin (mean={margin.mean():.3f})")

    ax = fig.add_subplot(gs[4, 3])
    mean_sims_per_class = [correct_sims[y_test == c].mean() for c in range(n_classes)]
    ax.bar(range(n_classes), mean_sims_per_class, color="tab:blue")
    ax.set_xticks(range(n_classes))
    ax.set_xlabel("True Class")
    ax.set_ylabel("Mean Dot Product")
    ax.set_title("Mean Correct-Class Dot Prod by Digit")

    fig_path = output_dir / "concat_diagnostics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostics to {fig_path}")


if __name__ == "__main__":
    train_and_evaluate()
