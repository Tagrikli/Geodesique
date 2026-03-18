# LOO Sparsity & Inactive Template Learning

## Problem
Inactive templates (a_i = 0) never learn — they're frozen in place by the active gating on the Rodrigues rotation. This wastes capacity since templates that start in unhelpful regions of the sphere can never move toward useful features.

## Attempt 1: Remove the active gating
Let all templates learn (rotate) on every input, not just active ones.

**Result:** All templates collapsed. Half converged to a single "average" direction, the other half to its antipode.

**Why it failed:** Inactive templates all share the same LOO shadow (x_recon, since a_i = 0 for all of them). Same shadow means same residual, same learning signal — they herd together. The antipodal split comes from templates on opposite hemispheres receiving tangent projections that point to opposite attractors.

## Attempt 2: Full-similarity reconstruction for learning
Include negative cosine similarities in the learning reconstruction (x_recon_full = s @ w instead of a @ w), so that each template — active or not — has a unique LOO shadow because s_i differs per template. Output activations still use ReLU.

**Result:** Still collapsed, just slightly slower.

**Why it failed:** With 25 templates, removing one template's contribution from the full reconstruction is a ~4% perturbation. The residual is ~96% shared signal, ~4% unique. The shared component dominates and still herds templates together.

## Key insight: LOO strength depends on reconstruction sparsity
In the original gated version, active templates didn't collapse because ReLU gating made the reconstruction sparse — only a few templates contributed. Removing one of 5 active templates is a 20% perturbation, giving a strong unique signal. But with ~50% of templates active (which is what ReLU gives on a high-dimensional sphere), LOO strength is only ~2/n_templates — it degrades linearly with template count. This also explains why learning slowed down with larger template counts.

## Proposed solution: k-Winners-Take-All
Replace ReLU gating with fixed top-k activation. Only the k most similar templates are active per input.

- LOO perturbation is always 1/k, independent of total template count
- k directly controls LOO signal strength (k=3 → 33% perturbation per removal)
- Scales to any number of templates without degrading learning
- Inactive templates remain frozen (preserving diversity) but the active set is always small enough for strong LOO signals
