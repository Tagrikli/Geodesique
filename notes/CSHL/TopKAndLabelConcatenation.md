# Top-K Selection and Label Concatenation

## Notes from experiments — 2026-03-28

---

## From Partial Correlations to Top-1

### The Pearson Correlation Discovery

When both input and templates are mean-centered unit vectors, the dot product is exactly the Pearson correlation. Before mean-centering templates, the output was contaminated by alignment with the mean direction — a component shared across all inputs that carries no structural information. Mean-centering both sides isolates the purely relational signal: which dimensions deviate above or below average, and does the pattern of deviations match?

This realization motivated mean-centering templates, not just inputs. The result felt cleaner because it *is* cleaner — the dot product now measures pure structural similarity with no dead weight from the mean component.

### Trying Partial/Semipartial Correlations as Inhibition

The idea was that competition between minicolumns computes something like partial or semipartial correlations — each template's unique contribution after factoring out what others explain. This would determine who learns from what.

Result: diversification was acceptable, but reconstruction quality was poor. The partial correlation machinery suppresses shared variance, which is exactly what you don't want if the goal is faithful reconstruction — the shared part is real structure, not noise.

### Moving to Top-1

Instead of soft inhibition or partial correlations, the next experiment used hard winner-take-all: only the template with the highest correlation learns from each input.

**Expected problem:** Dictatorship — one template dominates all inputs.

**Actual result:** No dictatorship. Because both input and templates are mean-centered and unit-length, there is no shared dominant direction to exploit. A template tuned to a 45° line has low correlation with a 90° line — they are genuinely different directions on the zero-mean unit sphere. The dictator needs a center of mass to camp near; mean-centering eliminates it.

This is the simplest possible competitive learning: winner-take-all + geodesic rotation toward input. Essentially online spherical k-means on the zero-mean hypersphere.

---

## Discretization of the Input Manifold

### Rotating Line Experiment

Input: 28x28 animation of a rotating line (width 5 pixels), mean-centered and L2-normalized. With top-1 selection, only ~5 templates learned distinct orientations of the line.

Each template captures a range of rotation angles where it outputs correlation above roughly 0.5. Beyond that threshold, a neighboring template wins. The templates form a Voronoi tessellation on the hypersphere — each owns the region of the manifold closest to it.

The resolution (number of distinct orientations captured) is controlled by the number of templates relative to the intrinsic dimensionality of the input manifold. A 1D manifold (rotation angle) gets tiled by ~5 templates with the default template count.

### Why Discretization Happens

A mean-centered line at 45° and one at 50° are very similar (high correlation). At 45° vs 90°, they're very different. Each template naturally covers a contiguous arc of the rotation manifold. The boundaries between arcs are where two templates have equal correlation with the input — the Voronoi cell edges.

---

## Label Concatenation

### The Experiment

MNIST digits were presented with a one-hot class label concatenated to the image vector. The combined vector was mean-centered and L2-normalized as usual.

Without the label: some templates fired for both visually similar digits (e.g., 5 and 6), since they have high structural correlation.

With the label concatenated: the system allocated separate templates for confusable digits. A single template cannot have high correlation with both "image of 5 + one-hot for 5" and "image of 6 + one-hot for 6" because the one-hot components are orthogonal. The geometry of the augmented space forces discrimination.

### Why This Works

The one-hot label adds dimensions where same-class inputs are identical and different-class inputs are orthogonal. In the augmented space, visually similar but categorically distinct inputs are pulled apart on the hypersphere. The top-1 competition then naturally allocates separate templates — no special learning rule, no loss function, no gradient. The discrimination pressure comes entirely from the input geometry.

### Inference Without Labels

When the one-hot dimensions are absent (or zeroed) at inference, the template with the highest visual correlation still tends to win. The templates learned visual structure alongside the label structure. The label amplified between-class separation during training, but the visual components remain discriminative. The margin is smaller, but the ranking is preserved.

### Generation From Labels Alone

When only the label is provided (image dimensions zeroed), the winning template's weight vector in the image dimensions serves as the reconstruction. The template *is* the learned prototype for that class. Same weights used for recognition and generation.

### The One-Template-Per-Digit Problem

With full one-hot labels, only one template is allocated per digit class. This happens because the one-hot encoding has zero within-class variation — every 5 gets the same label. The label dimensions dominate the correlation (perfectly consistent, high magnitude within class), so one template aligns to "average 5 + 5-label" and wins every 5. There is no within-class pressure to allocate a second template.

This is not a model failure — the model is doing exactly what the input geometry demands. The label has between-class discriminative power but zero within-class discriminative power. To get multiple templates per class, the concatenated signal would need to carry within-class distinctions (e.g., a richer encoding) or the label's influence would need to be scaled down so the visual dimensions can break ties within a class.

---

## Sparsity Effects at the Second Layer

### Setup

Layer 1: 4x4 kernel convolution on 28x28 MNIST with 36 templates, producing a (7,7,36) output. Top-1 selection means each spatial position has exactly one nonzero activation out of 36 channels.

Layer 2: 7x7 kernel with 25 templates, receiving the full (7,7,36) output as a single flattened 1764-dim vector. Of those 1764 dimensions, only ~49 are nonzero (one per spatial position).

### Sparsity Increases Selectivity

At Layer 1 (dense pixel input), a single template can dominate an entire digit class because the raw pixel correlation between two different instances of the same digit is high — there's enough shared dense structure for one template to match them all.

At Layer 2, the same digits have lower mutual correlation. Two different 5s that looked very similar in pixel space may have different L1 winners at some spatial positions. In the sparse L2 input, a mismatch at even one position means different dimensions are nonzero — and different nonzero dimensions contribute nothing to the dot product. The correlation drops.

The result: a single L2 template can no longer capture all instances of a digit. The system is forced to allocate multiple templates per digit class, each specializing on a sub-variant — a specific configuration of L1 winners across the spatial grid.

### Sparsity Does What Labels Couldn't

The one-template-per-digit problem at L1 (with label concatenation) arose because the label had zero within-class variation. The sparse L1 code solves this without any label: it introduces within-class variation by discretizing each spatial position independently. Two 5s that differ in one local patch produce different sparse codes, forcing L2 to either tolerate the mismatch (lower correlation) or allocate a second template.

The hierarchy naturally increases selectivity pressure through the representation change alone — no extra parameters, no feedback, no labels.

### Correlation Magnitude and Reconstruction Quality

When an L2 template achieves high correlation with its input, its reconstruction is faithful — the specific sparse configuration is well captured. When correlation is low, the reconstruction is poor. High L2 correlation means the template matches the exact constellation of L1 winners, which is a much more specific claim than matching a dense pixel pattern. The reconstruction quality at L2 is therefore a direct measure of how well the hierarchy has tiled the space of L1 activation configurations.
