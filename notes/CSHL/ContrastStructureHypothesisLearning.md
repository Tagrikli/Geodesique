# Contrast-Structure Hypothesis Learning (CSHL)

## Notes from a brainstorming session — established ideas, design decisions, and open questions.

---

## Core Idea

A set of **templates** (unit vectors) that learn online from input data. Templates live on the unit hypersphere and learn by **rotating** toward their learning targets. The system is inspired by cortical minicolumns within a hypercolumn.

Each template proposes a **hypothesis about the entire input** — not a part of it. Templates within a hypercolumn are competing hypotheses, like orientation-tuned minicolumns where one says "this is a 45° edge" and another says "this is a 30° edge." They don't collectively explain the input — each one tries to capture the whole thing.

---

## Why Unit Norm?

By constraining templates to unit length, we separate **structure** from **magnitude**. The input's norm carries scale information; the normalized input carries the pattern of relationships across dimensions. Templates only care about the latter.

This parallels **synaptic scaling** in biology — neurons maintain total synaptic weight within a range. They redistribute weight across synapses rather than growing unboundedly. A template can't inflate its weights to win more activations. It has to commit: putting more weight in one direction means taking it away from another. Learning is about how to spend a fixed budget.

---

## Mean Centering: Operating in the Realm of Pure Contrast

### The Decision

Both input and templates are **mean-centered then L2-normalized**. This places everything on the intersection of the unit sphere and the zero-mean hyperplane — a (d-1)-dimensional manifold.

### Why Center?

- **Centering removes overall level.** For pixel patches, this strips brightness and keeps contrast. A bright patch with a subtle edge and a dark patch with the same edge become identical after centering.
- **What remains is purely relational** — which dimensions are higher or lower *relative to each other*. No absolute values, only deviations from the mean.
- **Cosine similarity between two zero-mean unit vectors is Pearson correlation.** So template matching becomes pure correlation — "does the pattern of deviations match?"
- **Without centering, template mean is dead weight.** If the input is centered but the template isn't, the template's mean component contributes nothing to the dot product (because centered input is orthogonal to the all-ones vector). The mean just wastes norm budget, reducing sensitivity and creating functionally redundant templates that differ only in their invisible mean.

### Centering at Every Level

Each hypercolumn mean-centers its input independently. At the pixel level, this removes brightness. At higher levels, the input is a vector of activations from below — centering removes overall activation level but preserves the relative pattern of which features fired more or less than others. The semantics shift from "how much each feature is present" to "which feature is present relative to average."

This is recursive. At every level: strip out "how much total," keep "what's the relative pattern." The whole system speaks one language — correlation with contrastive structure.

### The Ganzfeld Argument

A uniform black patch and a uniform white patch both become zero vectors after centering. In the system, they're identical — no structure, nothing for templates to match. But we don't perceive them as identical. The resolution: you never see pure uniform anything. Structure lives at boundaries between regions, not within them. Patches at edges of uniform regions see transitions. The **Ganzfeld effect** confirms this — a truly uniform visual field causes perception to dissolve. The system needs contrast to function, and so does the brain.

### Deeper Philosophical Point

Everything in perception might be relative. You don't see absolute luminance — you see contrast. You don't hear absolute loudness — you hear changes. You don't evaluate "good" absolutely — you evaluate relative to expectation (reward prediction error). Mean centering at every level may reflect something fundamental about biological computation: every stage adapts away the baseline and passes forward only deviations.

---

## Mean Centering Resolves the Dictator Problem

In earlier experiments (WW2: Rotational Update with Winner-Take-All), a single "dictator" template would win almost every competition and collapse the population. This happened because raw pixel inputs all live in the positive orthant — every image has a shared "bright pixels" bias. A template that drifts toward the average positive direction wins everything, learns from everything, and becomes even more dominant. Classic rich-get-richer collapse.

Mean centering removes this failure mode at the root. After subtracting the mean, there is no dominant direction shared across the dataset. Each input's contrast pattern points in a genuinely different direction on the sphere. A "3" and a "7" have very different contrast signatures — one can't camp near both. No single template position is systematically closer to all inputs, because the mean of all centered inputs is approximately zero (the dataset has no directional bias in contrast space).

The dictator needed a "center of mass" to exploit. Mean centering eliminates it.

---

## Why Not Angular Separation Between Templates?

### The Problem

An early idea was to use cosine similarity between templates to prevent collapse. But in high dimensions, angular separation is a **poor proxy for functional distinctness**.

### The High-Dimensional Geometry Argument

In d=1000, random unit vectors are approximately orthogonal (expected cosine similarity ~0, std ~1/√d). Two templates can be orthogonal while both responding strongly to the same data cluster. Orthogonality says nothing about their relationship to a third vector (the data).

### The Manifold Argument

Real data lives on a thin manifold — a low-dimensional surface embedded in high-dimensional space. If data varies along 50 directions out of 1000, two templates could differ entirely in the 950 "dead" dimensions where data has no variance. They'd be orthogonal in ambient space but functionally redundant from the data's perspective, because the data never explores the dimensions where they differ.

### The Right Criterion

**Functional distinctness must be measured in data space, not parameter space.** Two templates are redundant if their response profiles across data points are correlated, regardless of their angle to each other. Two templates are distinct if they respond to different inputs, regardless of how close they are in parameter space.

---

## Activation Function: Global Competition, Not Partial Matching

### The Problem with Cosine Similarity as Partial Match

Consider a template that perfectly captures the top half of an MNIST digit but is zero in the bottom half. Cosine similarity penalizes it because the input has energy in dimensions the template ignores. The template perfectly matches its part but gets a low score because cosine similarity is a **global** measure.

This led to exploring reconstruction-based activations, LOO residuals, and other schemes for measuring partial contribution. But this was the wrong direction.

### The Shift: Templates Are Whole-Input Hypotheses

The cortical analogy clarifies everything. In a hypercolumn, a minicolumn tuned to 45° isn't explaining "the 45° part" of the input. It's proposing **"the whole input is a 45° edge."** Templates are competing hypotheses about the entire input, not collaborators explaining parts.

So cosine similarity (correlation, after centering) is the right activation function after all. A template that only partially matches the input *should* get a lower score, because it's a worse hypothesis about the whole input. A cross presented to orientation-tuned templates yields low activations across multiple templates — honestly reporting that no single orientation hypothesis is a strong match.

---

## Inhibition: Scalar and Global

### Not Dimension-Wise Subtraction

Early thinking imagined inhibition as LOO reconstruction subtracted dimension-by-dimension — each template sees only what others don't explain. This corresponds to collaborative decomposition.

### Scalar Broadcast Instead

Since templates are competing whole-input hypotheses, inhibition is **scalar**. A strongly active minicolumn broadcasts a global inhibitory signal: "I've got this, back off." Others are suppressed proportionally to the winner's activation, not dimension-by-dimension.

### Biological Consistency

The literature confirms: **lateral inhibition between minicolumns** is the primary mechanism within hypercolumns. Minicolumns do not inhibit themselves (self-inhibition exists as a separate, slower mechanism like adaptation). Inhibitory connections between minicolumns within a hypercolumn suppress all but those that best match the current input, facilitating self-organization of receptive fields.

### Iterative Convergence

The input is fixed. Activations evolve across iterations:

1. All minicolumns compute raw cosine similarities (correlations) with the input — initial excitation.
2. Each minicolumn receives inhibition proportional to the others' activations.
3. New activations are computed under inhibition. Weak responders get suppressed, strong ones amplified.
4. Repeat until the activation pattern converges to a stable equilibrium.

This produces a **soft winner-take-all** — the converged activation pattern is the output, not the raw similarities. The equilibrium reflects each minicolumn's competitive standing.

---

## Learning

At convergence, each minicolumn has a final activation reflecting its competitive success for this input. The winning (or strongly active) minicolumn **rotates toward the input** via Rodrigues rotation.

Over time, across many inputs, minicolumns specialize: each wins for a different subset of inputs and tunes itself to that subset. Common patterns attract dedicated templates (winning often = strong learning signal). Rare patterns are represented by weak distributed activations across multiple templates.

### Why the Dictator Problem Doesn't Apply Here

In WW2, raw (non-centered) inputs shared a dominant direction in the positive orthant. A single template could exploit this by sitting near the dataset's center of mass and winning nearly every competition. Mean centering removes this exploit: there is no shared dominant direction in contrast space. Each input's contrast pattern is structurally distinct, so no single template position can dominate across diverse inputs. Statistical diversity in the training data is sufficient to drive template specialization when the trivial shared component (the mean) has been removed.

### Rotation on the Constrained Manifold

Templates are zero-mean and unit-norm. Rodrigues rotation between two zero-mean vectors stays in the zero-mean hyperplane, because the rotation occurs in the plane spanned by two zero-mean vectors, and any linear combination of zero-mean vectors is zero-mean.

**Initialization:** Sample random unit vectors, subtract mean, renormalize. In high dimensions, random vectors are already approximately zero-mean, so the correction is small.

**Ongoing updates:** As long as both the template and learning target are zero-mean, rotation preserves the constraint automatically. No projection step needed (except occasional correction for floating-point drift).

---

## Hierarchical Architecture

### The Resolution Problem

A single hypercolumn with ~100 templates can't tile a high-dimensional data distribution. But it doesn't need to — it only answers one narrow question.

### Convolutional Hierarchy

**Level 1:** Small receptive fields (e.g., 4x4 pixel patches). Very few structurally distinct patterns exist at this scale — edges, corners, gradients, flat regions. A small set of templates (20-30) can tile this space. Each proposes "this patch is a [pattern type]." Winner-take-all competition picks the best match. Convolution across the image produces a sparse activation map of local structure.

**Level 2:** A hypercolumn receives the sparse activation map as input. Its templates are hypotheses about *configurations* of local features. "Horizontal edge on top, vertical edge on left" = a larger-scale corner. Competition picks the best structural hypothesis.

**Level N:** Each level's sparse output becomes the next level's input. Templates at each level are simple competing hypotheses, but inputs get progressively more abstract.

### Density-Aware Allocation Emerges Naturally

Common patterns earn dedicated templates (they win often, templates get reinforced). Rare patterns get only weak partial matches. The activation magnitude carries information: high activation = well-explained by a single hypothesis; low activation = novel or blended input that no template captures well.

The number of templates per hypercolumn is a **resolution parameter**: more templates = finer distinctions, fewer = coarser coverage of only dominant patterns.

### Hypercolumn Specialization

Different hypercolumns don't need to learn to attend to different aspects. At lower levels, they literally receive different inputs (different spatial locations via convolution). At higher levels, different hypercolumns receive outputs from different lower-level regions. Specialization comes from architecture, not learned routing.

---

## Summary of the Pipeline

For each hypercolumn at any level:

1. **Receive input** (pixel patch, or activation vector from below)
2. **Mean-center** the input (subtract mean — removes overall level, keeps relative structure)
3. **L2-normalize** (places input on the zero-mean unit sphere)
4. **Compute correlations** with all templates (cosine similarity = Pearson correlation, since both sides are zero-mean unit vectors)
5. **Iterative competition** — scalar lateral inhibition between minicolumns until activations converge
6. **Output** the converged activation pattern (sparse, with winning template having highest activation)
7. **Learn** — winning/active templates rotate toward the input on the constrained manifold

---

## Open Questions

- **How many iterations for convergence?** Is convergence guaranteed? Can it oscillate?
- **Inhibition strength** — what's the right scaling for the scalar inhibition? Too strong = hard winner-take-all, too weak = no specialization.
- **Self-inhibition as adaptation** — should templates have a separate fatigue mechanism (EMA of recent activity dampening future responsiveness)?
- **What does the system output for truly novel inputs?** All-low activations are informative ("I don't know this"), but how should downstream layers handle that signal?
- **Sparse vs. dense output** — after competition, is the output typically one strong winner, or a soft distribution? This depends on inhibition strength and could be a tunable parameter.
- **Mean information** — the subtracted mean at each level is discarded. Should it be routed somewhere as a separate signal (analogous to separate luminance/contrast pathways)?
- **Optimal number of templates per hypercolumn at different levels?**
- **How exactly does the convolutional step work?** Standard sliding window? Stride? Overlap?
