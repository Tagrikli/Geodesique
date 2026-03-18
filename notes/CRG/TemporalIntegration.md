# Temporal Integration of Input

We added a temporal integration buffer to `ContrastResidualGeodesic` that runs before any preprocessing (mean-centering and L2 normalization). It always runs regardless of the `is_learning` toggle, and resets alongside the weight matrix on `Reset`.

## Options Considered

**Standard EMA (weighted average)**
```
buffer = (1-γ) * x + γ * buffer
```
The current input gets weight `1-γ`, history gets `γ`. At `γ = 0` it reduces to pure input passthrough (no history). As `γ` approaches `1.0` the buffer integrates over a longer window. This is the correct model for membrane potential integration in a leaky neuron — the current input is always vivid, with past inputs fading beneath it at a rate controlled by `γ`.

**Additive decay (leaky accumulator)**
```
buffer = clip(x + γ * buffer, 0, 1)
```
Current input is added at full strength on top of decaying history — closest to retinal afterimages. Saturates if the input is sustained, which makes it less suitable as a general-purpose input buffer.

## Decision

We went with **Standard EMA**. At `γ = 0` it reduces to pure input passthrough (no history). As `γ` approaches `1.0` the integration window lengthens. The slider is exposed as **Trail Decay (γ)** with a log scale from `0.0` to `0.999`.
