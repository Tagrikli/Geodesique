"""Standalone benchmark: InputDigitMNIST → ContrastResidualGeodesic.

Replicates the exact init/process logic of both nodes as connected in
tutorial3.json, runs for 10 seconds, and reports loop speed in Hz.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/home/tagrikli/Desktop/Codes/MiniCortex")
from axonforge.nodes.utilities.dataset_cache import load_dataset_with_python_mnist


# ── InputDigitMNIST ──────────────────────────────────────────────────────────

def mnist_init():
    images, labels = load_dataset_with_python_mnist("mnist")
    perm = np.random.permutation(len(images))
    images = images[perm]
    labels = labels[perm]
    class_indices = {}
    for d in range(10):
        class_indices[d] = np.where(labels == d)[0]
    return {
        "images": images,
        "labels": labels,
        "class_indices": class_indices,
        "idx": 0,
        "repeat_counter": 0,
    }


def mnist_process(state, digit_filter=-1, repeat=1):
    images = state["images"]
    labels = state["labels"]

    rep = int(repeat)
    if rep <= 0:
        i = int(state["idx"])
        return images[i], int(labels[i])

    state["repeat_counter"] += 1
    if state["repeat_counter"] >= rep:
        state["repeat_counter"] = 0
        filt = int(digit_filter)
        if filt == -1:
            state["idx"] = (int(state["idx"]) + 1) % len(images)
        else:
            indices = state["class_indices"].get(filt, [])
            if len(indices) > 0:
                pos = np.searchsorted(indices, int(state["idx"]))
                pos = (pos + 1) % len(indices)
                state["idx"] = int(indices[pos])

    i = int(state["idx"])
    return images[i], int(labels[i])


# ── ContrastResidualGeodesic ──────────────────────────────────────────────────

def crg_init(amount=25, dim=784):
    w = np.random.randn(amount, dim)
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
    return {"w": w, "trail_buffer": None}


def crg_process(state, x_input, step_fraction=0.02, trail_decay=0.0, is_learning=True):
    eps = 1e-8

    w = np.asarray(state["w"], dtype=np.float64)
    x = np.asarray(x_input, dtype=np.float64).ravel()

    # Max-decay temporal integration (runs regardless of is_learning)
    gamma = float(trail_decay)
    if state["trail_buffer"] is None:
        state["trail_buffer"] = x.copy()
    else:
        state["trail_buffer"] = np.maximum(x, gamma * state["trail_buffer"])
    x = state["trail_buffer"]

    # Renormalize templates
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + eps)

    # 1. Mean-center and L2-normalize input
    x_mean = np.mean(x)
    x_c = x - x_mean
    x_c_norm = np.linalg.norm(x_c)
    if x_c_norm < eps:
        return

    x_hat = x_c / x_c_norm

    # 2. Activations
    s = w @ x_hat
    a = np.maximum(s, 0.0)

    # 3. Reconstruction
    x_recon = a @ w

    # 4. LOO shadow & residuals
    x_shadow = x_recon[None, :] - (a[:, None] * w)
    x_shadow_norm = np.linalg.norm(x_shadow, axis=1, keepdims=True)
    x_shadow_hat = x_shadow / (x_shadow_norm + eps)
    r_loos = x_hat[None, :] - x_shadow_hat

    # 5. Learning
    if is_learning:
        eta = float(step_fraction)
        r_dot_w = np.sum(r_loos * w, axis=1, keepdims=True)
        tau = r_loos - r_dot_w * w
        tau_norm = np.linalg.norm(tau, axis=1, keepdims=True)
        tau_hat = np.where(tau_norm > eps, tau / tau_norm, 0.0)
        active = a > 0
        theta = eta * tau_norm
        w_rotated = w * np.cos(theta) + tau_hat * np.sin(theta)
        w_new = np.where(active[:, None], w_rotated, w)
        w_new = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
        state["w"] = w_new

    # 6. LOO-inhibited activations
    shadow_overlap = np.maximum(np.sum(w * x_shadow, axis=1), 0.0)
    a_inh = np.maximum(s - shadow_overlap, 0.0)

    a_raw_out = a * x_c_norm
    a_inh_out = a_inh * x_c_norm
    x_mean_out = np.array(x_mean)

    return a_raw_out, a_inh_out, x_mean_out


# ── Main loop ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    mnist_state = mnist_init()
    crg_state = crg_init(amount=1024, dim=784)
    print("Dataset loaded. Running loop for 10 seconds...\n")

    duration = 10.0
    tick_times = []
    start = time.perf_counter()  # clock starts here, after all loading

    while True:
        t0 = time.perf_counter()
        if t0 - start >= duration:
            break

        # InputDigitMNIST.process()
        x, label = mnist_process(mnist_state, digit_filter=-1, repeat=1)

        # ContrastResidualGeodesic.process()
        crg_process(crg_state, x, step_fraction=0.029, trail_decay=0.0, is_learning=True)

        t1 = time.perf_counter()
        tick_times.append(t1 - t0)

    total_ticks = len(tick_times)
    elapsed = time.perf_counter() - start
    avg_hz = total_ticks / elapsed
    avg_ms = (sum(tick_times) / total_ticks) * 1000
    min_ms = min(tick_times) * 1000
    max_ms = max(tick_times) * 1000

    print(f"Ticks:        {total_ticks}")
    print(f"Elapsed:      {elapsed:.2f}s")
    print(f"Avg speed:    {avg_hz:.1f} Hz")
    print(f"Avg tick:     {avg_ms:.3f} ms")
    print(f"Min tick:     {min_ms:.3f} ms")
    print(f"Max tick:     {max_ms:.3f} ms")
