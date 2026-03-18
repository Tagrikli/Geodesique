"""Learning nodes for MiniCortex hypercolumn."""

import numpy as np

from axonforge.core.node import Node
from axonforge.core.descriptors import branch
from axonforge.core.descriptors.actions import Action
from axonforge.core.descriptors.displays import Heatmap
from axonforge.core.descriptors.ports import InputPort, OutputPort
from axonforge.core.descriptors.fields import Float, Integer, Range
from .utilities import to_display_grid


def _random_unit_weights(rows: int, cols: int) -> np.ndarray:
    w = np.random.randn(rows, cols)
    norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    return w / norms


class _SelfContainedLearningNode(Node):
    amount = Integer("Amount", default=25, on_change="_on_weight_shape_change")
    dim = Integer("Dim", default=784, on_change="_on_weight_shape_change")

    output = OutputPort("Weights", np.ndarray)

    weights_preview = Heatmap("Weights", colormap="bwr")

    reset = Action("Reset", lambda self, params=None: self._on_reset(params))

    def init(self):
        self._on_reset()

    def _on_reset(self, params=None):
        self._w = _random_unit_weights(int(self.amount), int(self.dim))
        self._publish_weights()
        return {"status": "ok"}

    def _on_weight_shape_change(self, new_value, old_value):
        if new_value != old_value:
            self._on_reset()

    def _publish_weights(self):
        self.output = self._w
        self.weights_preview = to_display_grid(self._w)


@branch("Hypercolumn/Learning")
class GeodesicHebbian(_SelfContainedLearningNode):
    inputs = InputPort("Input", np.ndarray)
    activations = InputPort("Activations", np.ndarray)

    alpha = Float("Alpha", default=1.0)

    activations_preview = Heatmap("Activations", colormap="grayscale")

    def process(self):
        if self.activations is not None:
            self.activations_preview = to_display_grid(
                np.asarray(self.activations).ravel()
            )

        if self.inputs is None or self.activations is None:
            self._publish_weights()
            return

        eps = 1e-8
        w = np.asarray(self._w, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()
        s = np.asarray(self.activations, dtype=np.float64).ravel()

        x_norm = x / (np.linalg.norm(x) + eps)

        cos_theta = np.clip(w @ x_norm, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        frac = np.clip(float(self.alpha) * s, 0.0, 1.0)
        delta = frac * theta

        sin_theta = np.sin(theta) + eps

        w_new = (
            (np.sin(theta - delta) / sin_theta)[:, None] * w
            + (np.sin(delta) / sin_theta)[:, None] * x_norm[None, :]
        )

        self._w = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
        self._publish_weights()


@branch("Hypercolumn/Learning")
class GeodesicHebbianResidual(_SelfContainedLearningNode):
    inputs = InputPort("Input", np.ndarray)
    activations = InputPort("Activations", np.ndarray)

    alpha = Float("Alpha", default=1.0)

    activations_preview = Heatmap("Activations", colormap="grayscale")
    residual_preview = Heatmap("Residual", colormap="bwr")

    def process(self):
        if self.activations is not None:
            self.activations_preview = to_display_grid(
                np.asarray(self.activations).ravel()
            )

        if self.inputs is None or self.activations is None:
            self._publish_weights()
            return

        eps = 1e-8
        w = np.asarray(self._w, dtype=np.float32)
        x = np.asarray(self.inputs, dtype=np.float32).ravel()
        s = np.asarray(self.activations, dtype=np.float32).ravel()

        # Keep templates unit-normalized.
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + eps)

        # s lives in normalized-input activation space, so decode there first.
        x_norm_factor = np.linalg.norm(x) + eps
        recon_hat = w.T @ s
        recon = recon_hat * x_norm_factor

        residual = x - recon
        self.residual_preview = to_display_grid(residual)
        res_norm = np.linalg.norm(residual)

        if res_norm < eps:
            self._w = w
            self._publish_weights()
            return

        r_hat = residual / res_norm

        cos_theta = np.clip(w @ r_hat, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        g = np.maximum(w @ r_hat, 0.0)

        frac = np.clip(float(self.alpha) * g, 0.0, 1.0)
        delta = frac * theta

        sin_theta = np.sin(theta)
        sin_theta = np.where(sin_theta < eps, eps, sin_theta)

        w_new = (
            (np.sin(theta - delta) / sin_theta)[:, None] * w
            + (np.sin(delta) / sin_theta)[:, None] * r_hat[None, :]
        )

        self._w = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
        self._publish_weights()


def geodesic_tangent(
    w_from: np.ndarray, w_to: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """
    Return tangent vector at w_from pointing toward w_to on unit sphere.

    The tangent is the component of (w_to - w_from) orthogonal to w_from,
    normalized to unit length.

    Args:
        w_from: Unit vector on sphere (starting point)
        w_to: Unit vector on sphere (target point)
        eps: Small constant for numerical stability

    Returns:
        Unit tangent vector at w_from pointing toward w_to
    """
    cos_theta = np.dot(w_from, w_to)
    tangent = w_to - cos_theta * w_from

    norm = np.linalg.norm(tangent)
    if norm < eps:
        return np.zeros_like(w_from)
    return tangent / norm


def rodrigues_rotation(
    w: np.ndarray, axis: np.ndarray, angle: float, eps: float = 1e-8
) -> np.ndarray:
    """
    Rotate vector w around axis by angle using Rodrigues' rotation formula.

    Assumes w and axis are already normalized unit vectors, and axis is
    orthogonal to w (lies in tangent space at w).

    Args:
        w: Unit vector to rotate
        axis: Unit rotation axis (should be orthogonal to w)
        angle: Rotation angle in radians
        eps: Small constant for numerical stability

    Returns:
        Rotated unit vector
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    if w.ndim == 1:
        return w * cos_a + axis * sin_a

    return w * cos_a[:, None] + axis * sin_a[:, None]


@branch("Hypercolumn/Learning")
class GeodesicPushPull(_SelfContainedLearningNode):
    """
    Force-directed competitive learning on the unit hypersphere.

    Combines data-driven attraction with constant repulsion between templates
    to achieve manifold coverage. Templates tile the data manifold like Voronoi
    cells, producing sparse, selective activations.
    """

    inputs = InputPort("Input", np.ndarray)

    alpha = Float("Alpha", default=1.0)
    beta = Float("Beta", default=1.0)
    step_fraction = Range(
        "Step Fraction", default=0.1, min_val=0.0, max_val=1.0, step=0.01
    )

    activations_preview = Heatmap("Activations", colormap="grayscale")

    def process(self):
        eps = 1e-8

        if self.inputs is None:
            self._publish_weights()
            return

        alpha = float(self.alpha)
        beta = float(self.beta)
        step_fraction = float(self.step_fraction)

        w = np.asarray(self._w, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()

        n_templates, _ = w.shape

        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self._publish_weights()
            return
        x_normalized = x / x_norm

        w_norms = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / (w_norms + eps)

        similarities = w @ x_normalized
        attraction_weights = np.maximum(similarities, 0.0)
        self.activations_preview = to_display_grid(attraction_weights)

        attraction_tangents = np.zeros_like(w)
        for i in range(n_templates):
            if attraction_weights[i] > eps:
                tangent = geodesic_tangent(w[i], x_normalized, eps)
                attraction_tangents[i] = attraction_weights[i] * tangent

        repulsion_tangents = np.zeros_like(w)
        for i in range(n_templates):
            for j in range(n_templates):
                if i == j:
                    continue

                cosine_ij = np.dot(w[i], w[j])
                if cosine_ij > 0:
                    tangent = geodesic_tangent(w[i], w[j], eps)
                    repulsion_tangents[i] += cosine_ij * tangent

        w_new = np.zeros_like(w)
        for i in range(n_templates):
            target_vec = w[i] + alpha * attraction_tangents[i] - beta * repulsion_tangents[i]
            target_norm = np.linalg.norm(target_vec)

            if target_norm < eps:
                w_new[i] = w[i]
                continue

            target_vec = target_vec / target_norm

            cos_angle = np.clip(np.dot(w[i], target_vec), -1.0, 1.0)
            angle_to_target = np.arccos(cos_angle)

            if angle_to_target < eps:
                w_new[i] = w[i]
                continue

            actual_rotation = step_fraction * angle_to_target
            rotation_axis = geodesic_tangent(w[i], target_vec, eps)
            w_new[i] = rodrigues_rotation(w[i], rotation_axis, actual_rotation, eps)

        self._w = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
        self._publish_weights()


@branch("Hypercolumn/Learning")
class GeodesicFunctionalPushPull(_SelfContainedLearningNode):
    """
    Functional competitive learning on the unit hypersphere.

    Uses inhibited activations and inhibition signals to drive learning,
    moving from structural (weight-space) repulsion to functional
    (activation-space) repulsion.
    """

    inputs = InputPort("Input", np.ndarray)
    activations = InputPort("Activations", np.ndarray)
    inhibition = InputPort("Inhibition", np.ndarray)

    alpha = Float("Alpha", default=1.0)
    beta = Float("Beta", default=1.0)
    step_fraction = Range(
        "Step Fraction", default=0.1, min_val=0.0, max_val=1.0, step=0.01
    )

    activations_preview = Heatmap("Activations", colormap="grayscale")
    inhibition_preview = Heatmap("Inhibition", colormap="grayscale")

    def process(self):
        eps = 1e-8

        if self.activations is not None:
            self.activations_preview = to_display_grid(
                np.asarray(self.activations).ravel()
            )

        if self.inputs is None or self.activations is None:
            self._publish_weights()
            return

        alpha = float(self.alpha)
        beta = float(self.beta)
        step_fraction = float(self.step_fraction)

        w = np.asarray(self._w, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()
        s_prime = np.asarray(self.activations, dtype=np.float64).ravel()

        if self.inhibition is not None:
            inh = np.asarray(self.inhibition, dtype=np.float64).ravel()
        else:
            inh = np.zeros_like(s_prime)
        self.inhibition_preview = to_display_grid(inh)

        n_templates, _ = w.shape

        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self._publish_weights()
            return
        x_hat = x / x_norm

        w_norms = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / (w_norms + eps)

        w_new = np.zeros_like(w)
        for i in range(n_templates):
            tau_hat_i = geodesic_tangent(w[i], x_hat, eps)
            f_i = alpha * s_prime[i] - beta * inh[i]
            tau_net = f_i * tau_hat_i

            v_target = w[i] + tau_net
            v_target_norm = np.linalg.norm(v_target)

            if v_target_norm < eps:
                w_new[i] = w[i]
                continue

            v_target = v_target / v_target_norm

            cos_theta = np.clip(np.dot(w[i], v_target), -1.0, 1.0)
            theta = np.arccos(cos_theta)

            if theta < eps:
                w_new[i] = w[i]
                continue

            actual_rotation = step_fraction * theta
            rotation_axis = geodesic_tangent(w[i], v_target, eps)
            w_new[i] = rodrigues_rotation(w[i], rotation_axis, actual_rotation, eps)

        self._w = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
        self._publish_weights()


@branch("Hypercolumn/Learning")
class GeodesicSoftmaxPushPull(_SelfContainedLearningNode):
    """
    Soft competitive learning on the unit hypersphere using Softmax probabilities.

    Uses Softmax probabilities to drive both attraction and repulsion,
    ensuring manifold coverage without additive biases.
    """

    inputs = InputPort("Input", np.ndarray)
    activations = InputPort("Activations", np.ndarray)

    alpha = Float("Alpha", default=1.0)
    beta = Float("Beta", default=1.0)
    step_fraction = Range(
        "Step Fraction", default=0.1, min_val=0.0, max_val=1.0, step=0.01
    )

    activations_preview = Heatmap("Activations", colormap="grayscale")

    def process(self):
        eps = 1e-8

        if self.activations is not None:
            self.activations_preview = to_display_grid(
                np.asarray(self.activations).ravel()
            )

        if self.inputs is None or self.activations is None:
            self._publish_weights()
            return

        alpha = float(self.alpha)
        beta = float(self.beta)
        step_fraction = float(self.step_fraction)

        w = np.asarray(self._w, dtype=np.float64)
        x = np.asarray(self.inputs, dtype=np.float64).ravel()
        p = np.asarray(self.activations, dtype=np.float64).ravel()

        n_templates, _ = w.shape

        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            self._publish_weights()
            return
        x_hat = x / x_norm

        w_norms = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / (w_norms + eps)

        w_new = np.zeros_like(w)
        for i in range(n_templates):
            tau_hat_i = geodesic_tangent(w[i], x_hat, eps)
            f_i = alpha * p[i] - beta * (1.0 - p[i])
            tau_net = f_i * tau_hat_i

            v_target = w[i] + tau_net
            v_target_norm = np.linalg.norm(v_target)

            if v_target_norm < eps:
                w_new[i] = w[i]
                continue

            v_target = v_target / v_target_norm

            cos_theta = np.clip(np.dot(w[i], v_target), -1.0, 1.0)
            theta = np.arccos(cos_theta)

            if theta < eps:
                w_new[i] = w[i]
                continue

            actual_rotation = step_fraction * theta
            rotation_axis = geodesic_tangent(w[i], v_target, eps)
            w_new[i] = rodrigues_rotation(w[i], rotation_axis, actual_rotation, eps)

        self._w = w_new / (np.linalg.norm(w_new, axis=1, keepdims=True) + eps)
        self._publish_weights()
