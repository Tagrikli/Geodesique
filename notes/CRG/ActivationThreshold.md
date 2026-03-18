# Activation Threshold (Dimension-Invariant)

## Motivation
LOO signal strength is 1/k where k is the number of co-active templates. With standard ReLU (threshold=0), ~50% of templates are active regardless of dimension (cosine similarities concentrate around 0 on a high-dimensional sphere). This makes LOO too weak with many templates.

We need to control sparsity to keep LOO strong, but top-k selection is biologically implausible — there's no global sorting mechanism in neural circuits. Instead, raising the depolarization threshold is a local operation each neuron can perform independently, analogous to lateral inhibition in hypercolumns where active neurons suppress weaker ones.

## Implementation
Replaced `max(s_i, 0)` with `max(s_i - threshold, 0)` where:

    threshold = slider_value / √d

The `1/√d` factor is the dimension-invariance coefficient — it's the standard deviation of cosine similarity between random unit vectors on a d-sphere. So the slider operates in units of σ: slider=1 means "1σ above mean," slider=2 means "2σ," etc. Same meaning regardless of dimension.

Applied to both raw and inhibited output activations.

## Effect on learning
The thresholded `a` flows into reconstruction, LOO shadows, and the active gating. Higher threshold → fewer active templates → sparser reconstruction → stronger LOO perturbation per active template → sharper, more precise learning signals.
