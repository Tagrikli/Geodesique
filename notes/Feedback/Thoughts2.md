# Hierarchical Gain-Feedback for CRG --- Conceptual and Mathematical Specification

## Vision

This document describes a hierarchical feedback mechanism for Contrast-Residual Geodesic (CRG) layers. The key idea: **feedback is a gain field on the input space**, computed by projecting a higher layer's activations through the lower layer's own weight matrix. This single mechanism simultaneously provides task-relevant learning pressure (during training) and contextual sharpening (during inference), and enables generation when run without input.

The architecture is, to our knowledge, the first algorithm that is simultaneously **local, online, single-sample, classifying, and generative** --- all from one mechanism.

---

## Conceptual Framework

### What Feedback Is Not

Feedback is not reconstruction. If L2 sends its reconstruction of L1's activations back to L1, it is telling L1 "here's what I think you said." But L1 already knows what it said --- it just said it. Echoing a smoothed version of L1's own output back to L1 is redundant. L1 has no use for it.

Feedback is not prediction of L1's activations. If L2 operates on a slower timescale (leaky integration), its reconstruction carries temporal context --- what L1 has *typically* been doing. But L2 already embodies that temporal integration in its own output to higher layers. Sending the prediction back down asks L1 to mimic what L2 already computed.

### What Feedback Is

Feedback is a **relevance signal** --- a gain field over L1's input dimensions that says "these parts of the input matter in the current context, these parts don't."

L1, operating alone, treats every input identically. It tiles the contrast manifold uniformly --- equal representational density everywhere. It has structure sensitivity but zero context sensitivity. It cannot know which of its detected features are task-relevant and which are noise.

L2 knows something L1 doesn't: which combinations of L1 features participate in coherent higher-order patterns. Feedback communicates this knowledge not by telling L1 what to output, but by shaping what L1 *sees*. The gain field amplifies task-relevant input dimensions and attenuates irrelevant ones. L1 then processes this shaped input through its normal CRG pipeline, honestly reporting what it finds.

### The Gain Field Couples Learning and Activation

The gain modulation multiplies the input *before* L1 processes it --- after mean-centering, before normalization. This means L1's templates never see the dimmed dimensions clearly. Because CRG's learning is driven entirely by the residual between input and LOO shadow, if part of the input is dimmed to near-zero, there is no residual there, no torque, no learning. The learning effect is not a separate mechanism --- it falls out of the activation effect for free.

This mirrors cortical coupling: the same dendritic activity that drives the soma's output is what drives plasticity. If the input to the dendrite is suppressed, both activation and learning are suppressed simultaneously. One mechanism, two effects.

### What the Gain Does to Manifold Tiling

Without feedback, L1 distributes its representational budget uniformly across the input manifold. With feedback, the gain field creates **non-uniform tiling pressure**. Templates that cover gain-amplified regions receive strong input, produce large activations, generate large residuals where not yet well-tuned, and undergo strong geodesic rotation. Templates covering dimmed regions get weak input, small activations, small residuals, and weak learning.

Over many training examples, L1's templates accumulate more structure in the amplified (task-relevant) regions of the manifold. The gain doesn't reorganize L1's templates directly --- it biases where L1 invests its plasticity. The manifold tiling becomes task-efficient: dense where discrimination is hard, sparse where it's easy.

### How L2's Hypothesis Is Tested

The gain field is a question L2 asks: "is this what I think it is?"

If the input matches L2's hypothesis, the gain amplifies the right features. L1's templates that detect those features activate crisply, LOO inhibition cleans up noise, and L1 sends up an activation vector consistent with what L2 expected. L2's templates fit the incoming activation well. L2's residuals are small. The system has settled.

If the input does not match L2's hypothesis, the gain still amplifies the hypothesis-relevant dimensions. But L1 processes what is actually there honestly. The actual contrast structure in those dimensions produces an activation pattern that looks nothing like what L2 expected. L2 receives this, its templates can't reconstruct it, and L2's residuals spike. L2 experiences surprise through its own normal CRG machinery --- no separate mismatch computation needed.

L1 never lies. It processes whatever gain-modulated input it receives through the same CRG pipeline regardless of L2's intentions. The gain is the question; L1's activation is the answer; L2's own reconstruction error is the evaluation.

### Temporal Context Without a Task

If L2 operates on a leaky integrator (exponential moving average of L1's activations), then without any task signal, L2's templates encode temporal regularities --- which L1 templates tend to co-fire over recent history. The gain field in this regime provides temporal smoothing: "the recent past looked like this, so bias toward those input dimensions."

This is useful for natural input streams with temporal coherence but doesn't create task-relevant sculpting. Every manifold region gets equal treatment over time --- the gain just tracks wherever the input stream happens to be. The temporal smoothing corrects for sequential continuity but doesn't prioritize any region over another.

### Supervised Feedback: The One-Hot Label

When a one-hot class label is clamped at the top of the hierarchy, the system becomes supervised. The one-hot selects one template at L_top. That template's relevance propagates down through each layer's weight matrices, becoming a progressively richer gain field at each level:

- At L_top: "template $k$ matters"
- One layer down: "the input features that template $k$ responds to matter"
- At L1: "these pixel-space dimensions are relevant to class $k$"

Each layer unpacks the one-hot through its own weights, transforming a maximally sparse signal into a structured, task-specific gain field over the raw input.

### Generation

The same gain-feedback pathway that does recognition does generation when input is absent.

Clamp a class label at the top. Propagate the gain down through the weight chain. The gain field that arrives at L1's input space is itself a generated image --- the network's learned answer to "what input structure corresponds to this class?" It is the chain of weight matrices projecting the one-hot all the way to pixel space.

No separate generative model. No adversarial training. No decoder. The recognition pathway run in reverse *is* the generator. The quality of the generated image is a direct diagnostic of what the hierarchy has learned.

### Self-Bootstrapping

At initialization, every layer's weights are random unit vectors. The gain field reaching L1 is the one-hot projected through random matrices --- random noise. This is no worse than no gain at all. L1 does approximately unsupervised CRG learning.

As weights develop structure, the gain becomes progressively more coherent and task-focused. The transition from unsupervised to supervised is smooth and automatic --- no pretraining, no phase switching, no curriculum design. The network bootstraps itself.

---

## Mathematical Formulation

### Setup

Consider a two-layer hierarchy for concreteness (the formulation extends to arbitrary depth).

- **L1 weight bank:** $W_1 \in \mathbb{R}^{N_1 \times D}$, each row $\mathbf{w}_{1,i}$ a unit vector. $D$ is raw input dimensionality (e.g., 784 for MNIST).
- **L2 weight bank:** $W_2 \in \mathbb{R}^{N_2 \times N_1}$, each row $\mathbf{w}_{2,j}$ a unit vector. L2's input space is L1's activation space.
- **Label feedback:** $\mathbf{y} \in \mathbb{R}^{N_2}$, a one-hot vector (or more generally, any top-down signal over L2's template indices).
- **Raw input:** $\mathbf{x} \in \mathbb{R}^D$.

### Step 1: Compute the Gain Field

The gain field is the label projected down through each layer's own weights.

**L2 gain (over L1's activation space):**

$$\mathbf{g}_2 = \mathbf{y}^\top W_2 \in \mathbb{R}^{N_1}$$

This is a vector over L1's template indices, encoding which L1 templates are relevant to the clamped class.

**L1 gain (over input space):**

$$\mathbf{g}_1^{\text{raw}} = \mathbf{g}_2^\top W_1 \in \mathbb{R}^{D}$$

This is a vector over input dimensions, encoding which input features are relevant to the clamped class, as mediated by L1's weight structure.

**Gain normalization:** The raw gain must be transformed into a multiplicative modulator centered around 1 (neutral). Several options exist; a simple one is shifting and scaling:

$$\mathbf{f} = 1 + \gamma \cdot \frac{\mathbf{g}_1^{\text{raw}} - \text{mean}(\mathbf{g}_1^{\text{raw}})}{\|\mathbf{g}_1^{\text{raw}} - \text{mean}(\mathbf{g}_1^{\text{raw}})\| + \epsilon}$$

Where $\gamma$ controls the gain strength. When $\gamma = 0$, feedback is neutral (all-ones). The mean-centering of the gain ensures it modulates contrast rather than adding a DC bias.

Alternative: a sigmoid or softmax mapping that keeps $\mathbf{f}$ strictly positive.

**For deeper hierarchies:** Each layer $\ell$ computes $\mathbf{g}_\ell = \mathbf{g}_{\ell+1}^\top W_\ell$, cascading from the top-most label down to the input.

### Step 2: Gain-Modulated Input to L1

Standard CRG mean-centers, then normalizes. The gain is applied between these steps:

$$\mathbf{x}_c = \mathbf{x} - \overline{x}$$

$$\mathbf{x}_g = \mathbf{f} \odot \mathbf{x}_c$$

$$\hat{\mathbf{x}} = \frac{\mathbf{x}_g}{\|\mathbf{x}_g\|}$$

The gain modulates the contrast signal. Subsequent L2 normalization redistributes the emphasis onto the unit sphere --- amplified dimensions occupy more of the unit vector's directional budget, dimmed dimensions occupy less.

### Step 3: L1 Forward Pass (Standard CRG)

L1 processes $\hat{\mathbf{x}}$ through its normal pipeline:

$$s_i = \mathbf{w}_{1,i} \cdot \hat{\mathbf{x}}, \qquad a_i = \max(s_i, 0)$$

$$\mathbf{x}_{\text{recon}} = \sum_j a_j \cdot \mathbf{w}_{1,j}$$

$$\mathbf{x}_{\text{shadow},i} = \mathbf{x}_{\text{recon}} - a_i \cdot \mathbf{w}_{1,i}$$

LOO-inhibited activations and learning proceed exactly as in single-layer CRG. No modification to the learning rule is needed --- the gain's effect on learning is mediated entirely through the shaped input.

### Step 4: L2 Forward Pass

L2 receives L1's LOO-inhibited activations $\mathbf{a}_{\text{inh}}$ (scaled by contrast norm) as its input. L2 processes this through its own CRG pipeline. If L2 uses a leaky integrator:

$$\mathbf{a}_{L2,\text{input}} = (1 - \alpha) \cdot \mathbf{a}_{L2,\text{avg}} + \alpha \cdot \mathbf{a}_{\text{inh}}$$

$$\mathbf{a}_{L2,\text{avg}} \leftarrow \mathbf{a}_{L2,\text{input}}$$

L2 then runs standard CRG on $\mathbf{a}_{L2,\text{input}}$.

### Step 5: Learning

Both layers update their weights through their own CRG residual dynamics. No backpropagation. No cross-layer gradient. Each layer learns from the residual between its (gain-shaped) input and its LOO shadow, using Rodrigues' rotation on the tangent plane.

The supervised signal affects L1's learning only through the gain: amplified dimensions produce larger residuals and stronger weight updates. The label never directly touches L1's weights.

---

## Generation Mode

To generate an image for class $k$:

1. Set $\mathbf{y} = \mathbf{e}_k$ (one-hot for class $k$).
2. Compute $\mathbf{g}_2 = \mathbf{e}_k^\top W_2$.
3. Compute $\mathbf{g}_1^{\text{raw}} = \mathbf{g}_2^\top W_1$.
4. The generated image is $\mathbf{g}_1^{\text{raw}}$ (or a normalized/rescaled version of it).

This is the hierarchy's learned answer to "what does class $k$ look like in input space?" No iterative sampling, no decoder network. The weight matrices used for classification are the same ones used for generation.

---

## Training Procedure

```
for each (image x, label y) in training set:

    # --- Compute gain field (top-down) ---
    g2 = y @ W2                              # (N1,) relevance over L1 templates
    g1_raw = g2 @ W1                         # (D,)  relevance over input dims
    f = gain_normalize(g1_raw, gamma)        # (D,)  multiplicative gain ≥ 0

    # --- L1 forward pass with gain ---
    x_c = x - mean(x)                       # mean-center
    x_g = f * x_c                            # apply gain
    x_hat = x_g / ||x_g||                   # normalize to sphere

    # ... standard CRG for L1 on x_hat ...
    # (activations, LOO shadow, inhibited output, weight update)

    # --- L2 forward pass ---
    a_L1 = L1_inhibited_activations          # L1's output
    # optionally: leaky integration
    # ... standard CRG for L2 on a_L1 ...
    # (activations, LOO shadow, inhibited output, weight update)
```

No pretraining phase. No separate unsupervised/supervised stages. Both layers learn simultaneously from the first sample. Early in training, gain is random noise (weights are random), so both layers do approximately unsupervised learning. As weights develop structure, the gain becomes coherent and task-relevant sculpting emerges gradually.

---

## Properties

### Locality

The gain field is computed from globally available signals (L2's activations, each layer's own weights), but the computation at each layer is local: each template only needs the gain-modulated input and the shared reconstruction.

### No Credit Assignment Problem

The label's effect on L1 is not an error signal propagated through the network. It is a gain modulation of the input. L1 learns from its own residuals on the shaped input. There is no chain rule, no gradient accumulation, no vanishing gradient.

### Unified Recognition and Generation

The same weight matrices, the same projection pathway, serve both functions. Recognition: gain × input → activations → classification. Generation: label → gain → read out as image. The generative quality is a free diagnostic of learned representation quality.

### Self-Scaling Curriculum

Early training: random weights → random gain → unsupervised learning. Late training: structured weights → coherent gain → supervised sculpting. The transition is continuous and requires no scheduling.

### Non-Uniform Manifold Tiling

The gain field concentrates learning on task-relevant manifold regions. Easy class distinctions (1 vs. 0) need minimal extra tiling. Hard distinctions (5 vs. 8) receive differential gain from their respective labels, driving finer template resolution exactly where it's needed.

---

## Biological Correspondence

| Mechanism | Cortical Analog |
|-----------|-----------------|
| Gain field on L1 input | Thalamic relay modulation (pulvinar, LGN) |
| L2 → gain projection through L1's weights | Cortico-thalamic feedback using L1's receptive field structure |
| Gain-coupled learning | Dendritic gating of plasticity (same activity drives output and LTP/LTD) |
| One-hot label at top | Task-specific prefrontal / executive signal |
| Leaky integration at L2 | Temporal integration in higher cortical areas (longer time constants) |
| L2 surprise (high residuals after mismatched gain) | Prediction error in higher areas driving orienting response |
| Generation via top-down gain without input | Mental imagery / top-down pattern completion |

---

## Design Decisions and Open Questions

### Gain normalization

The transformation from raw projection $\mathbf{g}_1^{\text{raw}}$ to multiplicative gain $\mathbf{f}$ needs to keep values positive and centered around 1. The specific nonlinearity (shift+scale, sigmoid, softmax) may affect learning dynamics. Empirical testing needed.

### Gain strength ($\gamma$)

Controls how much the feedback biases L1's input. Too weak and the task signal doesn't reach L1. Too strong and L1 only sees the gain field, ignoring the actual input. Optimal $\gamma$ may change during training (weak early when gain is noise, stronger later when gain is coherent). Could be adaptive.

### Leaky integration at L2

The time constant $\alpha$ determines how much temporal context L2 carries. For single-image classification, this may not be necessary. For sequence processing or temporal tasks, it becomes important. For the initial MNIST implementation, L2 can operate without leaky integration (pure feedforward on current L1 output).

### Depth

For MNIST (784-dim, 10 classes, simple manifold), two CRG layers plus a label interface likely suffice. More depth gives more levels of gain refinement but also more random projections to bootstrap through. For CIFAR or natural images, depth becomes more important.

### Ping-pong settling

The analysis-by-synthesis iterative loop (forward → feedback → reprocess → ...) can be added later. For the initial implementation, a single forward-feedback pass is sufficient: compute gain from label, process gain-modulated input through L1, send to L2, learn. The iterative settling refines ambiguous interpretations but is not necessary for the basic supervised learning loop.

### Stability of the gain-learning loop

The gain shapes what L1 learns. What L1 learns reshapes the gain (because the gain is projected through L1's weights). This is a coupled dynamical system. In principle, it could collapse (L1 tiles only the amplified region, gain becomes even more focused, positive feedback loop). In practice, the diversity of training labels should prevent this: the gain for "4" amplifies different dimensions than the gain for "7," so L1's templates are pulled in multiple directions. Empirical verification needed.