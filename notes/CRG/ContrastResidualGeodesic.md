# ContrastResidualGeodesic --- Contrast-Residual Geodesic Learning

## Vision

ContrastResidualGeodesic (CRG) models a cortical hypercolumn where **the LOO directional residual is the sole learning signal**. There are no separate pull/push forces and no balance parameter. The residual's sign naturally encodes both attraction (unclaimed features) and repulsion (over-explained features), and the sphere constraint redistributes weight budget automatically.

The key innovation is **mean-centering the input** before normalization. This extracts the *contrast* --- what distinguishes this particular pattern from the average --- and discards the DC component (mean brightness). On MNIST, this means learning focuses on digit *shape* rather than overall ink density.

---

## Mathematical Formulation

### Setup

- **Weight bank:** $W \in \mathbb{R}^{N \times D}$, where $N$ is the number of templates (minicolumns) and $D$ is the input dimensionality. Each row $\mathbf{w}_i$ is a unit vector on the hypersphere $S^{D-1}$: $\|\mathbf{w}_i\| = 1$.
- **Input:** $\mathbf{x} \in \mathbb{R}^D$ (raw pixel values, etc.).

### Step 1: Contrast Extraction and Normalization

$$\mathbf{x}_c = \mathbf{x} - \overline{x}$$

where $\overline{x} = \frac{1}{D}\sum_{d=1}^{D} x_d$ is the scalar mean across dimensions.

$$\hat{\mathbf{x}} = \frac{\mathbf{x}_c}{\|\mathbf{x}_c\|}$$

**Biological interpretation:** The mean-centering step models center-surround processing in early visual cortex (or retinal ganglion cells). It removes the DC component and normalizes the contrast signal, so all downstream computation operates on *what is distinctive* about this input. The templates learn contrast patterns rather than absolute intensity profiles. For positive-orthant data (like images), raw normalization preserves a strong bias toward the all-positive direction; mean-centering removes this bias.

### Step 2: Raw Similarities and Activations

$$s_i = \mathbf{w}_i \cdot \hat{\mathbf{x}}$$

$$a_i = \max(s_i, 0)$$

Standard cosine similarities with the contrast-normalized input, rectified. Because the input is mean-centered (and thus can have negative components even for image data), the distribution of similarities is more symmetric, and fewer templates are active compared to the raw-normalization baseline.

### Step 3: Global Reconstruction

$$\mathbf{x}_{\text{recon}} = \sum_{j=1}^{N} a_j \cdot \mathbf{w}_j = \mathbf{a}^\top W$$

The population's collective explanation of the contrast-normalized input.

### Step 4: LOO Shadow and Directional Residual

For each template $i$, remove its contribution:

$$\mathbf{x}_{\text{shadow},i} = \mathbf{x}_{\text{recon}} - a_i \cdot \mathbf{w}_i = \sum_{j \neq i} a_j \cdot \mathbf{w}_j$$

Normalize the shadow to unit length:

$$\hat{\mathbf{x}}_{\text{shadow},i} = \frac{\mathbf{x}_{\text{shadow},i}}{\|\mathbf{x}_{\text{shadow},i}\| + \epsilon}$$

Compute the **pure directional residual**:

$$\mathbf{r}_i = \hat{\mathbf{x}} - \hat{\mathbf{x}}_{\text{shadow},i}$$

**This residual is the central quantity.** Its direction and magnitude encode everything:

- **Where $\hat{\mathbf{x}}$ and $\hat{\mathbf{x}}_{\text{shadow},i}$ agree:** $\mathbf{r}_i \approx 0$. These features are already well-explained by the collective. No learning signal.
- **Where $\hat{\mathbf{x}}$ has structure that $\hat{\mathbf{x}}_{\text{shadow},i}$ lacks:** $\mathbf{r}_i$ points toward the unclaimed features. Attractive signal.
- **Where $\hat{\mathbf{x}}_{\text{shadow},i}$ has structure that $\hat{\mathbf{x}}$ lacks:** $\mathbf{r}_i$ points away from the over-explained features. Repulsive signal.

There is no separate pull/push decomposition. The residual naturally combines both forces.

### Step 5: LOO-Inhibited Activation (Soma Output)

The inhibited activation uses the **unnormalized** shadow:

$$\text{shadow\_overlap}_i = \max(\mathbf{w}_i \cdot \mathbf{x}_{\text{shadow},i}, \, 0)$$

$$a_{\text{inh},i} = \max(s_i - \text{shadow\_overlap}_i, \, 0)$$

The clamping of `shadow_overlap` to non-negative ensures that templates in the opposite hemisphere from the collective shadow are not artificially boosted.

The final output activations are scaled by the contrast norm:

$$a_{\text{out},i} = a_{\text{inh},i} \cdot \|\mathbf{x}_c\|$$

---

## Learning Rule

### Tangent Projection

The residual $\mathbf{r}_i$ is projected onto the tangent plane at $\mathbf{w}_i$:

$$\boldsymbol{\tau}_i = \mathbf{r}_i - (\mathbf{r}_i \cdot \mathbf{w}_i) \, \mathbf{w}_i$$

This removes the radial component, leaving only the component that can drive rotation on the sphere.

### Geodesic Rotation (Rodrigues' Formula)

Only **active** templates ($a_i > 0$) are updated:

$$\theta_i = \eta \cdot \|\boldsymbol{\tau}_i\|$$

$$\hat{\boldsymbol{\tau}}_i = \frac{\boldsymbol{\tau}_i}{\|\boldsymbol{\tau}_i\|}$$

$$\mathbf{w}_i^{\text{new}} = \mathbf{w}_i \cos(\theta_i) + \hat{\boldsymbol{\tau}}_i \sin(\theta_i)$$

$$\mathbf{w}_i^{\text{new}} \leftarrow \frac{\mathbf{w}_i^{\text{new}}}{\|\mathbf{w}_i^{\text{new}}\|}$$

Where $\eta$ is the step fraction (learning rate). The rotation angle $\theta_i$ is proportional to both the learning rate and the magnitude of the tangent vector (the "torque"). Larger residuals produce larger rotations. There is no separate pull/push tangent computation, no inhibition scaling, and no balance parameter. The residual tangent is the *only* learning signal, and inactive templates simply don't update.

---

## How the Residual Subsumes Pull and Push

The residual $\mathbf{r}_i = \hat{\mathbf{x}} - \hat{\mathbf{x}}_{\text{shadow},i}$ can be decomposed:

- The $+\hat{\mathbf{x}}$ component pulls the template toward the input (attraction).
- The $-\hat{\mathbf{x}}_{\text{shadow},i}$ component pushes the template away from what others explain (repulsion).

When projected onto the tangent plane, this naturally produces a net tangent that balances attraction and repulsion without any tunable ratio. The balance emerges from the geometry: if the shadow is close to the input (strong collective explanation), the residual is small and learning is slow; if the shadow is far from the input (weak collective explanation), the residual is large and learning is fast.

### Inactive Templates

Inactive templates ($a_i \leq 0$, meaning $s_i \leq 0$) simply don't update. The rationale is:

1. Mean-centering produces inputs with both positive and negative components, so templates are less likely to be permanently orthogonal to all inputs.
2. If a template isn't activated by a pattern, it shouldn't learn from that pattern.
3. Templates that are inactive for one input may become active for another, especially as other templates move and change the competitive landscape.

Inactive templates are preserved as-is, available for future inputs, rather than being randomly pushed around.

---

## How Sparsity Emerges

As templates specialize via the LOO residual learning rule:

1. Similar templates develop high mutual weight overlap ($\mathbf{w}_i \cdot \mathbf{w}_j$).
2. High overlap means strong shadow overlap for co-active templates.
3. Strong shadow overlap suppresses redundant activations via the LOO-inhibited output.
4. Only the best match among similar templates survives.

Mean-centering contributes additional sparsity: by removing the DC component, the input lives in a lower-dimensional subspace (the hyperplane orthogonal to the all-ones vector), which reduces the number of templates that can have positive similarity.

---

## Properties of the Algorithm

### Simplicity

CRG has few moving parts:
- One preprocessing step (mean-center + normalize)
- One learning signal (residual tangent)
- One tunable parameter ($\eta$)

The minimal parameter count makes the algorithm easier to reason about and removes sources of tuning sensitivity.

### Locality

Every computation for template $i$ depends only on:
- The input $\hat{\mathbf{x}}$ (feedforward)
- Template $i$'s own weights $\mathbf{w}_i$
- The shared population reconstruction $\mathbf{x}_{\text{recon}}$

The LOO shadow is computed as $\mathbf{x}_{\text{recon}} - a_i \cdot \mathbf{w}_i$ --- a local subtraction from the shared signal.

### Online Learning

Each sample is processed independently. Weights are updated immediately after each sample (no batching, no gradient accumulation).

### Manifold Preservation

All operations respect the unit hypersphere geometry:
- The residual is projected to the tangent plane at $\mathbf{w}_i$
- Weight updates use Rodrigues' rotation (geodesic transport)
- Renormalization ensures numerical stability

---

## Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `amount` | 25 | Number of templates (minicolumns) |
| `dim` | 784 | Dimensionality of each template |
| `step_fraction` ($\eta$) | 0.05 | Learning rate (geodesic rotation angle per unit torque) |
| `is_learning` | True | Enable/disable weight updates |

Note: **there is no balance parameter and no sparsity parameter**. The E/I balance emerges from the residual geometry, and sparsity emerges from the LOO shadow inhibition and learned weight geometry.

---

## Outputs

| Port | Content |
|------|---------|
| `weights` | Current weight matrix $W$ ($N \times D$) |
| `activations` | LOO-inhibited activations $\mathbf{a}_{\text{inh}} \cdot \|\mathbf{x}_c\|$ |
| `raw_activations` | Raw cosine activations $\mathbf{a} \cdot \|\mathbf{x}_c\|$ |

Both activation outputs are scaled by the contrast norm $\|\mathbf{x}_c\|$ to preserve signal strength.

---

## Biological Correspondence

| Algorithm Component | Cortical Analog |
|---------------------|-----------------|
| Mean-centering $\mathbf{x} - \overline{x}$ | Center-surround processing (retinal ganglion / LGN) |
| Template $\mathbf{w}_i$ | Minicolumn's synaptic weight profile |
| Raw similarity $s_i$ | Total dendritic excitation |
| LOO shadow $\mathbf{x}_{\text{shadow},i}$ | Inhibitory signal from basket cell network |
| Shadow overlap $\mathbf{w}_i \cdot \mathbf{x}_{\text{shadow},i}$ | Strength-weighted dendritic shunting/inhibition |
| LOO-inhibited activation $a_{\text{inh},i}$ | Soma firing rate after lateral inhibition |
| Directional residual $\mathbf{r}_i$ | Error signal at the dendrite: mismatch between input and collective |
| Tangent projection $\boldsymbol{\tau}_i$ | Eligibility trace filtered to the weight manifold |
| Step fraction $\eta$ | Plasticity rate (NMDA receptor efficacy) |

The removal of the balance parameter can be interpreted biologically: in CRG, the E/I balance is not set by a global neuromodulatory tone, but emerges locally from the residual between the input and the collective explanation. This is arguably more biologically realistic --- each minicolumn computes its own effective E/I balance based on its local competitive context.

---

## Pseudocode

```
function PROCESS(x, W, step_fraction, is_learning):
    # -- Setup --------------------------------------------------------
    N, D = shape(W)                        # N templates, D dimensions

    # -- Step 1: Contrast extraction and normalization ----------------
    x_c = x - mean(x)                      # mean-center (remove DC)
    if ||x_c|| ≈ 0: return                 # skip flat inputs
    x_hat = x_c / ||x_c||                  # unit contrast direction

    # Ensure all templates live on the unit sphere
    for i in 1..N:
        W[i] = W[i] / ||W[i]||

    # -- Step 2: Raw similarities & activations -----------------------
    for i in 1..N:
        s[i] = W[i] · x_hat                # cosine similarity
        a[i] = max(s[i], 0)                # rectified activation

    # -- Step 3: Global reconstruction --------------------------------
    x_recon = Σ_j  a[j] * W[j]            # shared population vector

    # -- Step 4: LOO shadow & directional residual --------------------
    for i in 1..N:
        x_shadow[i] = x_recon - a[i] * W[i]
        x_shadow_hat[i] = x_shadow[i] / ||x_shadow[i]||

        # Pure directional residual: what the input has that the
        # collective (without me) doesn't, and vice versa.
        r[i] = x_hat - x_shadow_hat[i]

    # -- Step 5: LOO-inhibited activations (output) -------------------
    for i in 1..N:
        shadow_overlap[i] = max(W[i] · x_shadow[i], 0)   # unnormalized, clamped
        a_inh[i] = max(s[i] - shadow_overlap[i], 0) * ||x_c||

    # Also output raw activations for comparison.
    for i in 1..N:
        a_raw[i] = a[i] * ||x_c||

    # ================================================================
    # LEARNING (weight update) --- only if is_learning is True
    # ================================================================
    if not is_learning: return W, a_inh, a_raw

    for i in 1..N:
        if a[i] ≤ 0: continue              # inactive templates don't update

        # -- Step 6: Tangent projection of residual -------------------
        # Remove the radial component (along W[i]) from the residual,
        # leaving only the component that drives rotation on the sphere.
        tau[i] = r[i] - (r[i] · W[i]) * W[i]

        if ||tau[i]|| ≈ 0: continue        # no learning signal

        # -- Step 7: Geodesic rotation (Rodrigues' formula) -----------
        theta = step_fraction * ||tau[i]||  # rotation angle (torque)
        tau_hat = tau[i] / ||tau[i]||       # rotation axis

        W[i] = W[i] * cos(theta) + tau_hat * sin(theta)
        W[i] = W[i] / ||W[i]||             # safety renormalize

    return W, a_inh, a_raw
```
