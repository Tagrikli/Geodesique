"""Evaluate feedback vs no-feedback SRL representations via linear classifiers.

Loads pre-trained two-layer SRL weights from a feedback1.json graph,
encodes MNIST/Fashion-MNIST through both paths (feedback-trained and
no-feedback-trained), trains logistic regression on the second layer's
LOO-inhibited activations, and produces comparative classification
metrics, confusion matrices, and plots.

No SRL learning happens here — weights are frozen, only used for
feedforward activation extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from mnist import MNIST
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-8

# ---- Configuration -------------------------------------------------------
GRAPH_PATH = "ax_graphs/feedback1.json"
DATASET = "digits"                  # "digits" or "fashion"
MAX_ITER = 1000                     # Max iterations for logistic regression solver
OUTPUT_DIR = ROOT / "experiments" / "feedback" / "r005_digits"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Graph parsing
# ---------------------------------------------------------------------------

def _load_graph_config(graph_path: Path) -> dict:
    """Parse feedback1.json and extract the two SRL paths.

    Returns a dict with keys 'feedback' and 'no_feedback', each containing:
        layer1_weights_path, layer2_weights_path,
        layer1_amount, layer1_dim, layer2_amount, layer2_dim
    """
    with graph_path.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    connections = graph["connections"]

    # Find which SRL nodes receive feedback input
    feedback_receivers = set()
    for conn in connections:
        if conn["to_input"] == "feedback":
            feedback_receivers.add(conn["to_node"])

    # Find SRL layer chains: LGN -> SRL_layer1 -> SRL_layer2
    lgn_targets = {}  # srl_id -> lgn_id
    srl_targets = {}  # srl_id -> upstream_srl_id
    for conn in connections:
        if conn["to_input"] == "inputs":
            src = conn["from_node"]
            dst = conn["to_node"]
            if nodes_by_id[src]["type"] == "LGN" and nodes_by_id[dst]["type"] == "SignedResidualLearning":
                lgn_targets[dst] = src
            elif nodes_by_id[src]["type"] == "SignedResidualLearning" and nodes_by_id[dst]["type"] == "SignedResidualLearning":
                srl_targets[dst] = src

    # Build paths: layer2 -> layer1 (via srl_targets), layer1 must connect from LGN
    paths = []
    for layer2_id, layer1_id in srl_targets.items():
        if layer1_id in lgn_targets:
            has_feedback = layer1_id in feedback_receivers
            paths.append((layer1_id, layer2_id, has_feedback))

    result = {}
    for layer1_id, layer2_id, has_feedback in paths:
        n1 = nodes_by_id[layer1_id]
        n2 = nodes_by_id[layer2_id]
        key = "feedback" if has_feedback else "no_feedback"
        result[key] = {
            "layer1_id": layer1_id,
            "layer2_id": layer2_id,
            "layer1_amount": n1["fields"]["amount"],
            "layer1_dim": n1["fields"]["dim"],
            "layer2_amount": n2["fields"]["amount"],
            "layer2_dim": n2["fields"]["dim"],
            "layer1_gain_strength": n1["fields"].get("gain_strength", 0.0),
            "layer2_gain_strength": n2["fields"].get("gain_strength", 0.0),
            "layer1_gain_sign_positive": n1["fields"].get("gain_sign_positive", False),
            "layer1_trail_decay": n1["fields"].get("trail_decay", 0.0),
            "layer2_trail_decay": n2["fields"].get("trail_decay", 0.0),
            "layer1_weights_file": n1["states"]["w"]["file"],
            "layer2_weights_file": n2["states"]["w"]["file"],
        }

    if "feedback" not in result or "no_feedback" not in result:
        raise ValueError(
            f"Expected both feedback and no-feedback paths in graph. Found: {list(result.keys())}"
        )

    # Extract input node presentation settings (repeat, blank_steps)
    input_node = None
    for n in graph["nodes"]:
        if n["type"].startswith("Input"):
            input_node = n
            break
    result["input"] = {
        "repeat": input_node["fields"].get("repeat", 1) if input_node else 1,
        "blank_steps": input_node["fields"].get("blank_steps", 0) if input_node else 0,
    }

    return result


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
    x_train = np.asarray(x_train, dtype=np.float32) / 255.0
    y_train = np.asarray(y_train)

    x_test, y_test = loader.load_testing()
    x_test = np.asarray(x_test, dtype=np.float32) / 255.0
    y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# SRL forward pass (frozen weights, no learning)
# ---------------------------------------------------------------------------

def _srl_forward_single_layer(x_hat: np.ndarray, x_norm, w: np.ndarray):
    """Single-layer SRL forward pass (signed, no ReLU).

    Parameters
    ----------
    x_hat : (D,) L2-normalized input
    x_norm : scalar, L2 norm of x before normalization
    w : (K, D) unit-norm weight matrix

    Returns
    -------
    a_inh : (K,) LOO-inhibited activations (scaled by x_norm)
    recon_err : scalar, reconstruction MSE
    s : (K,) raw cosine similarities (for e2e reconstruction)
    """
    # Signed cosine similarities
    s = w @ x_hat  # (K,)

    # Full signed reconstruction
    x_recon = s @ w  # (D,)

    # Reconstruction error (in original scale)
    recon_err = np.sum((x_recon * x_norm - x_hat * x_norm) ** 2) / x_hat.shape[0]

    # LOO shadow
    x_shadow = x_recon[None, :] - (s[:, None] * w)  # (K, D)
    shadow_overlap = np.sum(w * x_shadow, axis=1)  # (K,) signed

    # LOO-inhibited activations
    a_inh = (s - shadow_overlap) * x_norm  # (K,)

    return a_inh, recon_err, s


def _forward_two_layer_single(
    xi_lgn: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    feedback: np.ndarray | None,
    gain_strength_l1: float,
    gain_sign_positive_l1: bool = False,
    trail_decay_l1: float = 0.0,
    ema_buffer: np.ndarray | None = None,
    trail_decay_l2: float = 0.0,
    ema_buffer_l2: np.ndarray | None = None,
):
    """Single forward pass through two SRL layers with optional feedback gain and EMA.

    Parameters
    ----------
    xi_lgn : (D,) mean-centered input
    w1 : (K1, D) layer 1 weights
    w2 : (K2, K1) layer 2 weights
    feedback : (K1,) feedback signal from L2 (None on first step)
    gain_strength_l1 : gain modulation strength for layer 1
    gain_sign_positive_l1 : if True use +1 sign in gain formula, else -1
    trail_decay_l1 : EMA decay for L1 input (0.0 = no EMA)
    ema_buffer : running EMA state from previous step (None = initialize)

    Returns
    -------
    a_inh2 : (K2,) L2 inhibited activations
    err1 : L1 reconstruction MSE
    err2 : L2 reconstruction MSE
    s2 : (K2,) L2 raw cosine similarities
    feedback_out : (K1,) L2 reconstruction (feedback for next step)
    x_mod : (D,) gain-modulated input (for e2e recon reference)
    ema_buffer_out : updated EMA buffer (same shape as xi_lgn)
    """
    x = xi_lgn.copy()

    # Feedback gain modulation on L1 input
    if feedback is not None and gain_strength_l1 > 0:
        g_raw = feedback @ w1  # (D,) project feedback to pixel space
        sign = 1 if gain_sign_positive_l1 else -1
        gain = 1.0 + (gain_strength_l1 * g_raw) * sign
        x = gain * x

    # EMA temporal integration (matches learning4.py trail_decay logic)
    if trail_decay_l1 > 0.0:
        if ema_buffer is None:
            ema_buffer = x.copy()
        else:
            ema_buffer = (1.0 - trail_decay_l1) * x + trail_decay_l1 * ema_buffer
        x = ema_buffer

    # Layer 1
    x_norm1 = np.linalg.norm(x)
    if x_norm1 < EPS:
        k1, k2 = w1.shape[0], w2.shape[0]
        return np.zeros(k2), 0.0, 0.0, np.zeros(k2), np.zeros(k1), x, ema_buffer, ema_buffer_l2

    x_hat1 = x / x_norm1
    a_inh1, err1, _ = _srl_forward_single_layer(x_hat1, x_norm1, w1)

    # EMA temporal integration on L2 input
    if trail_decay_l2 > 0.0:
        if ema_buffer_l2 is None:
            ema_buffer_l2 = a_inh1.copy()
        else:
            ema_buffer_l2 = (1.0 - trail_decay_l2) * a_inh1 + trail_decay_l2 * ema_buffer_l2
        a_inh1 = ema_buffer_l2

    # Layer 2
    x_norm2 = np.linalg.norm(a_inh1)
    if x_norm2 < EPS:
        k2 = w2.shape[0]
        return np.zeros(k2), err1, 0.0, np.zeros(k2), np.zeros(w1.shape[0]), x, ema_buffer, ema_buffer_l2

    x_hat2 = a_inh1 / x_norm2
    a_inh2, err2, s2 = _srl_forward_single_layer(x_hat2, x_norm2, w2)

    # L2 feedback_out = reconstruction in L1 space (matches learning4.py line 182)
    feedback_out = s2 @ w2  # (K1,) unit-scale reconstruction

    return a_inh2, err1, err2, s2, feedback_out, x, ema_buffer, ema_buffer_l2


def encode_two_layer(
    x: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    gain_strength_l1: float = 0.0,
    gain_sign_positive_l1: bool = False,
    trail_decay_l1: float = 0.0,
    trail_decay_l2: float = 0.0,
    repeat: int = 1,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode samples through a two-layer SRL pipeline (LGN + SRL1 + SRL2).

    For feedback paths, runs `repeat` forward passes per sample with the
    feedback loop (L2 reconstruction -> L1 gain modulation), matching the
    training protocol. Takes the final step's activations.

    Parameters
    ----------
    x : (N, D) raw pixel vectors in [0, 1]
    w1 : (K1, D) first layer weights
    w2 : (K2, K1) second layer weights
    gain_strength_l1 : gain modulation strength for L1 (0.0 = no feedback effect)
    repeat : number of presentation steps per sample (feedback iterations)
    desc : progress bar description

    Returns
    -------
    activations : (N, K2) second-layer LOO-inhibited activations
    recon_err_l1 : (N,) per-sample reconstruction error at layer 1
    recon_err_l2 : (N,) per-sample reconstruction error at layer 2
    recon_err_e2e : (N,) end-to-end reconstruction error (L2 -> W2 -> W1 -> image)
    """
    n = x.shape[0]
    k2 = w2.shape[0]
    activations = np.empty((n, k2), dtype=np.float32)
    recon_err_l1 = np.empty(n, dtype=np.float32)
    recon_err_l2 = np.empty(n, dtype=np.float32)
    recon_err_e2e = np.empty(n, dtype=np.float32)

    for i in tqdm(range(n), desc=desc, leave=False):
        xi = x[i].astype(np.float64)

        # LGN: mean-center
        xi_lgn = xi - np.mean(xi)

        # Run `repeat` forward passes with feedback loop
        feedback = None
        ema_buf = None
        ema_buf_l2 = None
        a_inh2 = np.zeros(k2)
        err1, err2 = 0.0, 0.0
        s2 = np.zeros(k2)
        for _ in range(repeat):
            a_inh2, err1, err2, s2, feedback, _, ema_buf, ema_buf_l2 = _forward_two_layer_single(
                xi_lgn, w1, w2, feedback, gain_strength_l1, gain_sign_positive_l1,
                trail_decay_l1, ema_buf, trail_decay_l2, ema_buf_l2,
            )

        # Use final step's results
        recon_err_l1[i] = err1
        recon_err_l2[i] = err2
        activations[i] = a_inh2.astype(np.float32)

        # End-to-end reconstruction: L2 -> L1 space -> image space
        x_norm2 = np.linalg.norm(a_inh2)
        if x_norm2 < EPS:
            recon_err_e2e[i] = np.sum(xi_lgn ** 2) / xi_lgn.shape[0]
        else:
            l1_recon = (s2 @ w2) * x_norm2  # (K1,) back to L1 scale
            image_recon = l1_recon @ w1  # (D,) back to image space
            recon_err_e2e[i] = np.sum((image_recon - xi_lgn) ** 2) / xi_lgn.shape[0]

    return activations, recon_err_l1, recon_err_l2, recon_err_e2e


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate():
    graph_path = ROOT / GRAPH_PATH
    print(f"Loading graph config from {graph_path}")
    config = _load_graph_config(graph_path)

    for key in ("feedback", "no_feedback"):
        c = config[key]
        print(f"\n  [{key}]")
        print(f"    Layer 1: {c['layer1_id']}  amount={c['layer1_amount']}  dim={c['layer1_dim']}")
        print(f"    Layer 2: {c['layer2_id']}  amount={c['layer2_amount']}  dim={c['layer2_dim']}")
        print(f"    Weights: {c['layer1_weights_file']}, {c['layer2_weights_file']}")

    # Load weights
    weights = {}
    for key in ("feedback", "no_feedback"):
        c = config[key]
        w1 = np.load(ROOT / c["layer1_weights_file"]).astype(np.float64)
        w2 = np.load(ROOT / c["layer2_weights_file"]).astype(np.float64)

        # Validate shapes against graph config
        expected_shape1 = (c["layer1_amount"], c["layer1_dim"])
        expected_shape2 = (c["layer2_amount"], c["layer2_dim"])
        assert w1.shape == expected_shape1, f"{key} L1: expected {expected_shape1}, got {w1.shape}"
        assert w2.shape == expected_shape2, f"{key} L2: expected {expected_shape2}, got {w2.shape}"

        # Renormalize to unit norm
        w1 = w1 / (np.linalg.norm(w1, axis=1, keepdims=True) + EPS)
        w2 = w2 / (np.linalg.norm(w2, axis=1, keepdims=True) + EPS)
        weights[key] = (w1, w2)
        print(f"\n  [{key}] Loaded: L1={w1.shape}, L2={w2.shape}")

    # Load dataset
    print(f"\nLoading MNIST ({DATASET})...")
    x_train, y_train, x_test, y_test = _load_mnist(DATASET)
    print(f"  Train: {x_train.shape[0]}  Test: {x_test.shape[0]}")

    # Encode through both paths
    repeat = config["input"]["repeat"]
    results = {}
    for key in ("feedback", "no_feedback"):
        w1, w2 = weights[key]
        c = config[key]
        label = "with feedback" if key == "feedback" else "no feedback"

        # Feedback path: use gain modulation + repeat steps
        # No-feedback path: gain_strength=0, single step
        if key == "feedback":
            gs_l1 = c["layer1_gain_strength"]
            gsp_l1 = c["layer1_gain_sign_positive"]
            td_l1 = c["layer1_trail_decay"]
            td_l2 = c["layer2_trail_decay"]
            n_repeat = repeat
        else:
            gs_l1 = 0.0
            gsp_l1 = False
            td_l1 = 0.0
            td_l2 = 0.0
            n_repeat = 1

        print(f"\nEncoding train set ({label}, repeat={n_repeat}, gain={gs_l1}, sign_positive={gsp_l1}, trail_decay=({td_l1},{td_l2}))...")
        z_train, re_l1_train, re_l2_train, re_e2e_train = encode_two_layer(
            x_train, w1, w2, gain_strength_l1=gs_l1, gain_sign_positive_l1=gsp_l1,
            trail_decay_l1=td_l1, trail_decay_l2=td_l2, repeat=n_repeat, desc=f"train ({label})",
        )
        print(f"Encoding test set ({label}, repeat={n_repeat}, gain={gs_l1}, sign_positive={gsp_l1}, trail_decay=({td_l1},{td_l2}))...")
        z_test, re_l1_test, re_l2_test, re_e2e_test = encode_two_layer(
            x_test, w1, w2, gain_strength_l1=gs_l1, gain_sign_positive_l1=gsp_l1,
            trail_decay_l1=td_l1, trail_decay_l2=td_l2, repeat=n_repeat, desc=f"test ({label})",
        )

        # Train linear classifier
        print(f"Training logistic regression ({label})...")
        clf = LogisticRegression(
            max_iter=MAX_ITER,
            solver="lbfgs",
            C=1.0,
            verbose=1,
        )
        clf.fit(z_train, y_train)

        # Evaluate
        y_pred_train = clf.predict(z_train)
        y_pred_test = clf.predict(z_test)
        y_proba_test = clf.predict_proba(z_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        top3_acc = top_k_accuracy_score(y_test, y_proba_test, k=3)
        top5_acc = top_k_accuracy_score(y_test, y_proba_test, k=5)

        report_str = classification_report(y_test, y_pred_test, digits=4)
        report_dict = classification_report(y_test, y_pred_test, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_test)

        train_active_frac = float(np.mean(np.abs(z_train) > EPS))
        test_active_frac = float(np.mean(np.abs(z_test) > EPS))

        print(f"\n{'='*50}")
        print(f"  [{label.upper()}]")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test  accuracy: {test_acc:.4f}")
        print(f"  Top-3 accuracy: {top3_acc:.4f}")
        print(f"  Top-5 accuracy: {top5_acc:.4f}")
        print(f"  L1 recon MSE (test): {re_l1_test.mean():.6f}")
        print(f"  L2 recon MSE (test): {re_l2_test.mean():.6f}")
        print(f"  E2E recon MSE (test): {re_e2e_test.mean():.6f}")
        print(f"{'='*50}\n")
        print(report_str)

        results[key] = {
            "label": label,
            "z_train": z_train,
            "z_test": z_test,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "y_proba_test": y_proba_test,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
            "report_str": report_str,
            "report_dict": report_dict,
            "cm": cm,
            "train_active_frac": train_active_frac,
            "test_active_frac": test_active_frac,
            "recon_err_l1_train": float(re_l1_train.mean()),
            "recon_err_l2_train": float(re_l2_train.mean()),
            "recon_err_l1_test": float(re_l1_test.mean()),
            "recon_err_l2_test": float(re_l2_test.mean()),
            "recon_err_e2e_train": float(re_e2e_train.mean()),
            "recon_err_e2e_test": float(re_e2e_test.mean()),
            "recon_err_l1_test_arr": re_l1_test,
            "recon_err_l2_test_arr": re_l2_test,
            "recon_err_e2e_test_arr": re_e2e_test,
            "config": config[key],
        }

    # Create run directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun directory: {output_dir}")

    # Save graph config (network parameters) for reproducibility
    graph_config_path = output_dir / "graph_config.json"
    with graph_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved graph config to {graph_config_path}")

    # Save results JSON
    json_results = {}
    for key in ("feedback", "no_feedback"):
        r = results[key]
        json_results[key] = {
            "label": r["label"],
            "layer1_id": r["config"]["layer1_id"],
            "layer2_id": r["config"]["layer2_id"],
            "layer1_shape": [r["config"]["layer1_amount"], r["config"]["layer1_dim"]],
            "layer2_shape": [r["config"]["layer2_amount"], r["config"]["layer2_dim"]],
            "train_accuracy": r["train_acc"],
            "test_accuracy": r["test_acc"],
            "top3_accuracy": r["top3_acc"],
            "top5_accuracy": r["top5_acc"],
            "train_active_fraction": r["train_active_frac"],
            "test_active_fraction": r["test_active_frac"],
            "recon_err_l1_train": r["recon_err_l1_train"],
            "recon_err_l2_train": r["recon_err_l2_train"],
            "recon_err_l1_test": r["recon_err_l1_test"],
            "recon_err_l2_test": r["recon_err_l2_test"],
            "recon_err_e2e_train": r["recon_err_e2e_train"],
            "recon_err_e2e_test": r["recon_err_e2e_test"],
            "max_iter": MAX_ITER,
            "dataset": DATASET,
            "graph_path": GRAPH_PATH,
            "classification_report": r["report_dict"],
            "confusion_matrix": r["cm"].tolist(),
        }

    results_path = output_dir / "eval_feedback_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Plot comparative diagnostics
    visualize_comparison(results, y_test, output_dir)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(ax, cm, title: str):
    class_labels = list(range(10))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(class_labels)
    ax.set_yticks(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color=color)


def _plot_normalized_cm(ax, cm, title: str):
    class_labels = list(range(10))
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0.0)
    im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(class_labels)
    ax.set_yticks(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)


def visualize_comparison(results: dict, y_test: np.ndarray, output_dir: Path):
    class_labels = list(range(10))
    fb = results["feedback"]
    nf = results["no_feedback"]

    fig = plt.figure(figsize=(28, 24), constrained_layout=True)
    fig.suptitle(
        f"Feedback vs No-Feedback — SRL 2-Layer Linear Probe ({DATASET.upper()})  |  "
        f"FB acc: {fb['test_acc']:.4f}  vs  No-FB acc: {nf['test_acc']:.4f}",
        fontsize=15,
    )
    gs = GridSpec(4, 4, figure=fig)

    # Row 0: Confusion matrices (counts)
    ax = fig.add_subplot(gs[0, 0:2])
    _plot_confusion_matrix(ax, fb["cm"], "Feedback — Confusion Matrix (counts)")

    ax = fig.add_subplot(gs[0, 2:4])
    _plot_confusion_matrix(ax, nf["cm"], "No Feedback — Confusion Matrix (counts)")

    # Row 1: Normalized confusion matrices
    ax = fig.add_subplot(gs[1, 0:2])
    _plot_normalized_cm(ax, fb["cm"], "Feedback — Confusion Matrix (normalized)")

    ax = fig.add_subplot(gs[1, 2:4])
    _plot_normalized_cm(ax, nf["cm"], "No Feedback — Confusion Matrix (normalized)")

    # Row 2, col 0: Per-class F1 comparison
    ax = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(10)
    width = 0.35
    fb_f1 = [fb["report_dict"][str(c)]["f1-score"] for c in class_labels]
    nf_f1 = [nf["report_dict"][str(c)]["f1-score"] for c in class_labels]
    ax.bar(x_pos - width / 2, fb_f1, width, label="Feedback", color="tab:blue")
    ax.bar(x_pos + width / 2, nf_f1, width, label="No Feedback", color="tab:orange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # Row 2, col 1: Per-class precision comparison
    ax = fig.add_subplot(gs[2, 1])
    fb_prec = [fb["report_dict"][str(c)]["precision"] for c in class_labels]
    nf_prec = [nf["report_dict"][str(c)]["precision"] for c in class_labels]
    ax.bar(x_pos - width / 2, fb_prec, width, label="Feedback", color="tab:blue")
    ax.bar(x_pos + width / 2, nf_prec, width, label="No Feedback", color="tab:orange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Precision")
    ax.set_title("Per-Class Precision Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # Row 2, col 2: Per-class recall comparison
    ax = fig.add_subplot(gs[2, 2])
    fb_rec = [fb["report_dict"][str(c)]["recall"] for c in class_labels]
    nf_rec = [nf["report_dict"][str(c)]["recall"] for c in class_labels]
    ax.bar(x_pos - width / 2, fb_rec, width, label="Feedback", color="tab:blue")
    ax.bar(x_pos + width / 2, nf_rec, width, label="No Feedback", color="tab:orange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # Row 2, col 3: Summary accuracy bar chart
    ax = fig.add_subplot(gs[2, 3])
    metrics = ["Test Acc", "Top-3", "Top-5"]
    fb_vals = [fb["test_acc"], fb["top3_acc"], fb["top5_acc"]]
    nf_vals = [nf["test_acc"], nf["top3_acc"], nf["top5_acc"]]
    x_m = np.arange(len(metrics))
    ax.bar(x_m - width / 2, fb_vals, width, label="Feedback", color="tab:blue")
    ax.bar(x_m + width / 2, nf_vals, width, label="No Feedback", color="tab:orange")
    ax.set_xticks(x_m)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    for i, (fv, nv) in enumerate(zip(fb_vals, nf_vals)):
        ax.text(i - width / 2, fv + 0.01, f"{fv:.3f}", ha="center", fontsize=7)
        ax.text(i + width / 2, nv + 0.01, f"{nv:.3f}", ha="center", fontsize=7)

    # Row 3, col 0: Reconstruction error comparison (bar)
    ax = fig.add_subplot(gs[3, 0])
    re_labels = ["L1 Train", "L1 Test", "L2 Train", "L2 Test", "E2E Train", "E2E Test"]
    fb_re = [fb["recon_err_l1_train"], fb["recon_err_l1_test"],
             fb["recon_err_l2_train"], fb["recon_err_l2_test"],
             fb["recon_err_e2e_train"], fb["recon_err_e2e_test"]]
    nf_re = [nf["recon_err_l1_train"], nf["recon_err_l1_test"],
             nf["recon_err_l2_train"], nf["recon_err_l2_test"],
             nf["recon_err_e2e_train"], nf["recon_err_e2e_test"]]
    x_r = np.arange(len(re_labels))
    ax.bar(x_r - width / 2, fb_re, width, label="Feedback", color="tab:blue")
    ax.bar(x_r + width / 2, nf_re, width, label="No Feedback", color="tab:orange")
    ax.set_xticks(x_r)
    ax.set_xticklabels(re_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction Error")
    ax.legend(fontsize=8)

    # Row 3, col 1: Reconstruction error distribution (L1 test)
    ax = fig.add_subplot(gs[3, 1])
    ax.hist(fb["recon_err_l1_test_arr"], bins=80, alpha=0.6, label="Feedback", color="tab:blue")
    ax.hist(nf["recon_err_l1_test_arr"], bins=80, alpha=0.6, label="No Feedback", color="tab:orange")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.set_title("L1 Recon Error Distribution (test)")
    ax.legend(fontsize=8)

    # Row 3, col 2: E2E reconstruction error distribution (test)
    ax = fig.add_subplot(gs[3, 2])
    ax.hist(fb["recon_err_e2e_test_arr"], bins=80, alpha=0.6, label="Feedback", color="tab:blue")
    ax.hist(nf["recon_err_e2e_test_arr"], bins=80, alpha=0.6, label="No Feedback", color="tab:orange")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.set_title("E2E Recon Error Distribution (test)")
    ax.legend(fontsize=8)

    # Row 3, col 3: Active templates per sample distribution (L2)
    ax = fig.add_subplot(gs[3, 3])
    fb_active = np.sum(np.abs(fb["z_test"]) > EPS, axis=1)
    nf_active = np.sum(np.abs(nf["z_test"]) > EPS, axis=1)
    ax.hist(fb_active, bins=range(int(fb_active.max()) + 2), alpha=0.6,
            label=f"FB (mean={fb_active.mean():.1f})", color="tab:blue")
    ax.hist(nf_active, bins=range(int(nf_active.max()) + 2), alpha=0.6,
            label=f"No-FB (mean={nf_active.mean():.1f})", color="tab:orange")
    ax.set_xlabel("# Active templates (L2)")
    ax.set_ylabel("Count")
    ax.set_title("Active L2 Templates per Sample")
    ax.legend(fontsize=8)

    fig_path = output_dir / "eval_feedback_diagnostics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostics figure to {fig_path}")


if __name__ == "__main__":
    train_and_evaluate()
