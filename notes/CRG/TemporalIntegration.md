# Temporal Integration of Input

We added a temporal integration buffer to `ContrastResidualGeodesic` that runs before any preprocessing (mean-centering and L2 normalization). It always runs regardless of the `is_learning` toggle, and resets alongside the weight matrix on `Reset`.

## Options Considered

**Standard EMA (weighted average)**
```
buffer = α * x + (1-α) * buffer
```
The current input gets weight `α`, history gets `1-α`. At low `α`, history dominates and the current frame barely registers — the buffer converges toward a slowly-shifting mean. This is the correct model for membrane potential integration in a leaky neuron, but wrong for what we want here: the current input should always be vivid, with past inputs fading beneath it.

**Additive decay (leaky accumulator)**
```
buffer = clip(x + γ * buffer, 0, 1)
```
Current input is added at full strength on top of decaying history — closest to retinal afterimages. Saturates if the input is sustained, which makes it less suitable as a general-purpose input buffer.

**Max-decay (peak hold with fade)**
```
buffer = max(x, γ * buffer)   # element-wise
```
Wherever the current input is bright, it shows at full value. Wherever past inputs were bright but the current frame isn't, those values fade toward zero at rate `γ`. No saturation, no unbounded growth. Closest to how V1 complex cells pool peak responses over a short temporal window.

## Decision

We went with **max-decay**. At `γ = 0` it reduces to pure input passthrough (no history). As `γ` approaches `1.0` the trail lengthens. The slider is exposed as **Trail Decay (γ)** with a log scale from `0.0` to `0.999`.
