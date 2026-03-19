Here's a summary of everything we discussed.

---

## Summary of CRG Architecture and Feedback Design Discussion

### Starting Point

We began from a working single-layer CRG (Contrast-Residual Geodesic) algorithm that has been validated on MNIST and Fashion MNIST. The algorithm is clean: mean-centered input, LOO shadow for both inhibition and learning, a single residual tangent as the learning signal, geodesic rotation on the hypersphere. No balance parameter, no sparsity parameter. It works, doesn't collapse, and produces cross-domain transferable features. The goal of this conversation was to think through how feedback from higher layers should work.

### Feedback as Activation Gain

The first major idea was that feedback from a higher layer (L2) should modulate the **activation gain** of L1 templates, not modify the input or the learning rule. Feedback is a vector in L1's activation space — one value per template. Values above 1 boost a template, values below 1 suppress it, all-ones means no feedback (the feedforward-only baseline).

The critical insight was how this interacts with existing CRG mechanics: **boosting a template's activation changes the LOO shadow for every other template.** The boosted template contributes more to the reconstruction, which means other templates that overlap with it get stronger shadow inhibition. They're pushed away — not because feedback told them to move, but because the LOO mechanism treats the boosted template as having claimed that territory. Over time, the boosted template learns more (because it's more active), competing templates differentiate away, and the boosted template ends up winning on feedforward merit alone even without feedback. Feedback creates a temporary advantage that the LOO dynamics convert into permanent structural change. And this automatically shapes learning without needing a separate learning-bias mechanism — because in CRG, active templates learn, inactive ones don't. Gating activation gates learning.

### Moving Exponential Average at L2

L2 doesn't see individual L1 activations per stimulus. Instead, it operates on a moving exponential average of L1's activations. This means L2 learns persistent co-activation patterns — which L1 templates reliably fire together across recent inputs. This has several effects: it filters per-stimulus noise (only stable structure persists in the average), it creates a natural hierarchy of learning speeds (persistent features consolidate first), and it makes L2's feedback represent contextual expectation rather than per-stimulus response. When input is homogeneous (many digit 4s), feedback is content-specific. When input is heterogeneous, feedback is diffuse — a natural emergence of the distinction between content-directed attention and content-free arousal.

### Surprise Modulates Feedback Authority

The mismatch between L1's current activation and L2's prediction (derived from the moving average) is a natural surprise signal. High surprise should weaken feedback influence (push toward all-ones, letting feedforward competition dominate) and speed up the moving average (so L2 adapts faster to the new regime). Low surprise keeps feedback strong and the moving average slow. This creates an orienting response: surprise opens the system up, it processes the novel input without preconceptions, and feedback re-engages once the new regime stabilizes.

### Supervision as Feedback from a Non-Existent Layer

The one-hot label for classification is semantically identical to feedback from a higher layer. Every layer gets feedback from above — L2 sends predictions to L1, L3 sends to L2, and so on. The label is just the boundary condition: feedback from a hypothetical perfect layer above the final layer, one that has already solved the task. The same mechanism, same code path. The label boosts the correct template at the final layer, the LOO dynamics propagate this through the competition, and the whole hierarchy adjusts. Without labels, the system is fully unsupervised. With labels, it's supervised at the boundary but unsupervised internally. Labels can be provided at any layer for partial supervision.

### Reduced Learning Rate During Inference, Not Frozen Weights

Rather than freezing weights during the feedback ping-pong and then doing a separate learning step, we keep learning on with a reduced rate during settling. The label at the top is an immovable anchor. Each ping-pong iteration, the lower layers shift slightly to become more consistent with both the input from below and the label from above. The label doesn't change — it forces lower layers to accommodate it. This distributes the global constraint (correct classification) into local weight changes through the residual geometry, without backpropagation. Credit assignment happens through iterative settling, not gradient flow.

### Residual Surfacing: Amplify What Feedback Doesn't Explain

This was a key conceptual flip. The standard assumption is that lower layers amplify what matches the feedback (confirmation). The proposed alternative: **lower layers amplify what the feedback fails to explain** (disconfirmation). When L2 sends a prediction down, L1 surfaces the residual — the parts of the input that L2's prediction missed. This forces L2 to deal with evidence it wasn't accounting for.

This creates a convergent settling dynamic. First iteration: L1 has no feedback, activations are fuzzy ("I don't know what I'm looking at"). L2 responds weakly. It sends a tentative prediction. L1 surfaces what's unexplained. L2 adjusts. L1 surfaces the remaining unexplained parts. Each iteration, the unexplained residual shrinks. Convergence means the hierarchy is internally consistent — every layer's prediction accounts for what the layer below sees. The number of iterations to convergence is a natural measure of input difficulty.

With the fixed label at top, this becomes even more powerful. The label says "cat." The residual bounces off this immovable wall. Lower layers must reorganize to make their representations consistent with the cat interpretation AND the actual input. If they can (residuals shrink), the label was correct. If they can't (residuals stay large), the input is genuinely ambiguous or the label is wrong.

### Attention Coexists with Residual Surfacing

Initially there seemed to be a tension: if lower layers amplify what doesn't match feedback, how does attention work (which should focus on specific features)? The resolution: feedback biases which templates win the LOO competition (attention), and residual surfacing within the attended feature domain provides fine-grained detail (refinement). When feedback is broad and uncertain (settling phase), residuals are large and drive major hypothesis changes. When feedback is narrow and confident (after settling), residuals are small and drive refinement within the attended feature. Same mechanism, different regime.

### PFC as Metacognitive Monitor

A PFC-like module operates on moving averages from multiple layers at a slow timescale. It sees the oscillation dynamics during settling — which templates are stable vs. flickering, how quickly the system converges, where in the hierarchy the instability lives. It doesn't do object recognition. It manages the inference process: detecting when the system is stuck, deciding whether to suppress the current winning hypothesis, directing attention to contested features, adjusting feedback gain and moving average speed. The pattern of oscillation across layers encodes *why* the system is confused (which features are contested), not just *that* it's confused.

### The Pulvinar as the Cascading Reconstruction Hub

The final and possibly most architecturally significant idea. In the brain, the pulvinar thalamic nucleus is connected bidirectionally with every level of the visual hierarchy (V1, V2, V4, IT). It's organized with internal subdivisions that mirror the cortical hierarchy. This suggests that the pulvinar is where the **cascading reconstruction happens** — not in the cortex itself.

Each cortical layer sends its activations to the pulvinar. The pulvinar does the top-down reconstruction cascade internally: the highest subdivision reconstructs into the next, which reconstructs into the next, all the way down to the lowest subdivision and from there into input space (the LGN gate). This runs in parallel with cortical feedforward processing.

This solves several problems. The highest cortical layer doesn't need to know how to speak pixel-language — it sends its state to the pulvinar, and the pulvinar's internal weights handle the translation through each level. The reconstruction is fast because the pulvinar is dedicated to this job rather than also doing competitive feature detection. And the thalamic gating of raw input (attention in pixel space) is just the bottom of this same cascade, not a separate mechanism.

The separation of concerns is clean: cortical CRG layers learn **features** (what to detect, through competitive LOO dynamics). The pulvinar learns **relationships between levels** (how detections at different abstraction levels predict each other, through its own reconstruction weights). Each does what it's good at.

---

The overall architecture that emerged: CRG layers stacked in a hierarchy, each doing local competitive learning with LOO shadows and residual-based geodesic updates. A pulvinar-like hub doing cascading reconstruction from high-level interpretations down to low-level predictions and input-space gating. Moving exponential averages providing temporal context. Surprise modulating feedback authority and adaptation speed. Residual surfacing driving convergent settling. Labels entering as feedback from the top boundary. A PFC-like monitor managing the inference dynamics. All built on the same core mechanism — competitive templates on the hypersphere with LOO residuals as the universal learning and inference signal.