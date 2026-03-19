# Mean Centering as a Separate Computation Stage

## Separation of Concerns: LGN and Cortical Layers

Mean centering is now treated as a responsibility of a dedicated preprocessing stage (LGN) rather than an operation performed within each cortical layer. The cortical hypercolumn receives already-centered input and applies only L2 normalization to project onto the unit hypersphere.

This separation is motivated by the observation that mean centering naturally propagates through the representation without explicit re-centering at each layer.

## Self-Emerging Mean-Centered Output

When inhibited activations are computed as `s - shadow_overlap` without ReLU gating, the output is itself mean-centered. This is not engineered — it emerges from the geometry of the learned templates. Because the input distribution is centered at the origin, the templates learn a symmetric arrangement on the hypersphere. Their sum converges toward the zero vector, which guarantees that the mean of all signed activations is approximately zero for any input.

This means each layer's output is already in the same form as its input: mean-centered, ready for the next layer to L2-normalize and process. No explicit mean subtraction or mean forwarding is required between layers.

## Reinterpreting Sparsity

With ReLU gating, roughly 50% of activations are zeroed out, producing a sparse code. Without gating, this same 50% carries negative values — anti-correlations with templates that are genuinely informative. What appeared as sparsity was actually the suppression of half the signal.

The signed (non-ReLU) output may constitute a richer representation for downstream layers. The notion of sparsity becomes less clear-cut: the code is dense in the sense that all templates contribute, yet it remains structured by the LOO inhibition that suppresses redundant positive activations.

## Conceptual vs. Literal Implementation

Rather than attempting to replicate the literal firing patterns observed in biological neurons, this approach models the *effective signal* that is communicated between layers. The assumption is that what matters for downstream computation is the information content of the layer's output — not the exact mechanism by which it is transmitted.

This is a deliberate shift from mimicking biological implementation details to capturing the computational role at a conceptual level. Just as other parts of the algorithm abstract away biological specifics (geodesic learning on a hypersphere rather than literal synaptic dynamics), the output representation is chosen to reflect what is effectively transmitted rather than how it is physically encoded.
