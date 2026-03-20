# Universal Contrastive Principle — Exploration Notes

## Starting Point

These notes document an exploration that began from the working SRL (Signed Residual Learning) algorithm within the Geodésique project. The core question: if contrast is the fundamental operation at the pixel level and at the template level, does the same principle extend to categories, feedback, attention, motor control, and every other function the architecture needs?

The answer appears to be yes. What follows is the chain of reasoning that led there.

---

## The Contrastive Primitive

The architecture already uses contrast at two levels:

**Pixel-level contrast (LGN):** The raw input is mean-centered: $\mathbf{x}_c = \mathbf{x} - \bar{x}$. Positive values mean "more than average here," negative values mean "less than average here." The absolute level is discarded. L2 normalization then discards magnitude, keeping only the direction of the contrast. Templates learn pure structure — the shape of deviations, not their scale.

**Template-level contrast (LOO):** Each template's learning signal is the residual $\mathbf{r}_i = \text{sign}(s_i) \cdot (\hat{\mathbf{x}} - \hat{\mathbf{x}}_{\text{shadow},i})$. This encodes "what the input has that the collective (without me) doesn't explain." The template defines itself contrastively against its peers.

Both operations share the same formal structure: take a thing, subtract its context, learn from the deviation. The deviation is signed — it says both what is present and what is absent. The magnitude is discarded — only the structure of the deviation matters.

---

## Category as Contrast

A category is only meaningful in the presence of other categories. If there is only one category, there is no category. "This is a 4" is meaningless without the implicit "and not a 7, not a 9, not a 1." Category is defined by what something is *and what it is not* — it is a contrastive signal.

This is formally identical to pixel-level contrast. Just as pixel meaning is "deviation from the mean pixel," categorical meaning is "deviation from the alternative categories."

If we collect running activation means per class, then for class $k$ we have $\mu_k$ (what class $k$ looks like in activation space) and the mean of all other class means $\mu_{\neg k}$. The categorical contrast $\mu_k - \mu_{\neg k}$ is a signed vector: positive components indicate features reliably present in class $k$ and absent in others, negative components indicate the reverse, and near-zero components indicate shared, non-discriminative structure.

---

## Avoiding External Supervision

The goal is not to impose category structure from outside. Supervised signals — target activations, loss gradients, forced rotations — are unnatural forces that override the system's own dynamics. The system's learning rule (SRL) is already contrastive: every template follows its signed residual. Introducing a supervised rotation would replace this natural signal with an external one.

Instead, the goal is to **guide** the system's natural contrastive abilities using another contrastive signal — something more like dopamine than backpropagation. The system figures out the discriminative directions on its own; the guidance signal only says whether the current contrast is sufficient or insufficient.

---

## Categorical Contrast as Multiplicative Gain

The categorical contrast $\mu_k - \mu_{\neg k}$, projected to input space through the weight matrix, produces a gain field over input dimensions. This gain field is applied multiplicatively to the mean-centered input before L2 normalization:

$$\mathbf{x}_g = \mathbf{f} \odot \mathbf{x}_c$$

$$\hat{\mathbf{x}} = \frac{\mathbf{x}_g}{\|\mathbf{x}_g\|}$$

Multiplicative gain does not invent structure. It cannot create features where there are none — multiplying zero by any gain still gives zero. It only controls exposure: brightening some dimensions, dimming others. After L2 normalization, the brightened dimensions take up more of the unit vector's angular budget and the dimmed dimensions take up less.

Templates then learn from this spotlight-lit input through their normal SRL dynamics. Where the spotlight is bright, residuals are large, tangent vectors have more torque, templates rotate faster. Where it's dim, residuals are small, learning is slow. Over time, templates accumulate more structure in the categorically relevant regions of input space — not because they were told what features to learn, but because those features were made more visible.

This is the same operation as the LGN. The LGN dims "what's average" and brightens "what's distinctive about this input." The categorical gain dims "what's shared across classes" and brightens "what's distinctive about this class." Same operation, different level of abstraction.

---

## SRL Output Carries Both Invariance and Equivariance

In SRL, the LOO-inhibited activations and the signed activations are the same thing. The output $s_i - \text{shadow\_overlap}_i$ is already mean-centered (as shown in the MeanCenteringAndLayerCommunication note — this emerges from the geometry of learned templates, not from explicit centering).

This signed, mean-centered output simultaneously carries:

- **Invariant information:** Positive components indicate "these templates match, this structure is present." Negative components indicate "these templates anti-match, this structure is absent." Together they describe the identity of the input.
- **Equivariant information:** The specific activation magnitudes preserve instance-level detail — how strongly each template responds to this particular input.

When averaged across instances of class $k$, the invariant structure survives (consistently positive or consistently negative components) while the equivariant detail washes out (variable components average toward zero). The class mean $\mu_k$ is therefore the invariant signature of class $k$ — what's always there and what's never there.

The categorical contrast $\mu_k - \mu_{\neg k}$ amplifies both: features reliably present in $k$ and absent in others get strong positive components, features reliably absent in $k$ but present in others get strong negative components. The gain field is a signed spotlight that says "look here where the thing is" and "look here where the thing isn't."

---

## Feedback as Cascading Contrastive Concepts

The categorical contrast doesn't need to originate from explicit class labels. In a hierarchy, each layer can send its contrastive understanding downward as a gain field:

1. The top layer develops its own natural clusters through SRL tiling.
2. Whatever contrastive concept it forms — whether from supervised class means or unsupervised cluster structure — gets projected through its weight matrix into the layer below's activation space.
3. That projection becomes a gain field over the layer below's input.
4. The layer below processes gain-modulated input, sends activations up.
5. The process repeats at every level.

Each layer translates "the thing I see" — defined contrastively as what it is and what it isn't — into the language of the layer below. At each level the contrastive concept becomes richer and more spatially specific as it's unpacked through the weight chain.

Up and down speak the same language because there is only one language: signed, mean-centered contrastive deviation.

---

## Unsupervised Categories from Lossy Reconstruction

Without labels, the top layer still provides useful feedback. Its finite template capacity forces it to tile the activation manifold unevenly — dense coverage where many inputs cluster, sparse coverage where few inputs live. Its reconstruction of any given input is therefore biased: it reflects back what it has learned (the recurring patterns, the natural clusters) and distorts what it hasn't (the rare, idiosyncratic, between-cluster structure).

This lossy reconstruction, projected back down as a gain field, is already a proto-categorical signal. It amplifies structure that the top layer recognizes (belongs to a well-tiled cluster) and dims structure that the top layer can't capture (falls between clusters). Lower layers receiving this gain learn more from the recognized structure, gradually aligning their representations with the top layer's natural clustering.

Categories emerge without labels. The dopamine/supervision signal then becomes a way to *name* and *correct* clusters the system would have discovered anyway — "you grouped horse with dog, that's bad" — rather than a way to create categories from scratch.

---

## Compositional Self-Similarity

A single template defines itself contrastively against its LOO shadow. A single layer defines its representation contrastively through the competitive dynamics of all its templates. The hierarchy defines its categories contrastively through the lossy reconstruction cascading between layers.

The functional roles that emerge at different levels — feature detection at the bottom, pattern recognition in the middle, classification at the top — are not designed. They are consequences of the same operation applied to different input types. The bottom layer detects features because that's what contrastive tiling does to pixels. The top layer classifies because that's what contrastive tiling does to activation patterns that are themselves contrastively tiled.

What a single layer does, the network does compositionally.

---

## Hierarchical Efficiency

The hierarchy makes the whole network efficient at recognizing what it has seen before, just as the learning rule makes each layer efficient at recognizing what it has seen before. Same principle, different scale:

- A **template** rotates toward frequently encountered directions → becomes efficient for those directions.
- A **layer** distributes templates to cover populated manifold regions → becomes efficient for that data distribution.
- The **hierarchy**, through feedback gain fields, makes every layer efficient at recognizing patterns that the layers above have learned to expect → the whole system becomes efficient for the experienced world.

The gain fields from above are the system's prior. Familiar inputs get sharp processing and minimal learning (small residuals). Novel inputs trigger cascading incoherence throughout the hierarchy (large residuals everywhere) — a natural surprise signal that emerges from the same mechanism without a separate novelty detector.

---

## Spatiotemporal Composition via EMA

If each layer has an exponential moving average (EMA) with a progressively longer time constant, the contrastive signal at each level becomes "what is this *over this timescale* versus what it isn't *over this timescale*":

- **Bottom layer (fast EMA):** Near-instantaneous contrast. Templates tile the manifold of momentary patterns. Captures "what's happening right now."
- **Middle layers (medium EMA):** Stable feature combinations that survive temporal averaging. Captures "what's been happening."
- **Top layer (slow EMA):** Enduring structure, persistent categories. Captures "what generally happens."

Feedback reverses the timescale cascade: the top layer's gain says "based on what generally happens, attend to these features," which biases the middle layer toward "given what's been happening, these combinations matter," which biases the bottom layer toward "right now, look at this."

Temporal abstraction, prediction, and multi-scale surprise all emerge from composition of one operation across time as well as space.

### Limitation: EMA Loses Temporal Order

An EMA at a high layer with a long time constant blurs everything into a distribution. "Dog running" and "dog standing then running" produce similar averages. The actual sequential structure — *this* happened *then that* happened — is lost.

This motivates a recurrent loop at the top of the hierarchy: the top layer's output feeds into another layer, which feeds back. This loop learns the manifold of state transitions, not just the manifold of states. Its templates encode trajectory segments — "when the representation looks like this, it typically becomes that."

The EMA provides context (slow, orderless, like mood or familiarity). The recurrent loop provides narrative (ordered, directional, like sequential reasoning or planning). Both are needed.

When external input is removed and the loop runs autonomously, it generates plausible continuations of experience from learned trajectory dynamics. This is mental simulation — not replaying stored memories, but running the learned state-transition model forward.

---

## Cortical Universality

The architecture maps onto multiple cortical systems with the same core circuit.

### Visual Cortex
- **Input:** 2D retinotopic pixel array.
- **LGN:** Mean-centering extracts visual contrast.
- **Convolutional SRL layers:** Templates learn local spectral/spatial features, tile with progressive abstraction.
- **Feedback:** Higher layers send contrastive gain fields that spotlight task-relevant visual features.

### Auditory Cortex
- **Input:** 1D frequency array (cochlear decomposition). Each dimension is a frequency band, each value is energy in that band.
- **LGN equivalent:** Mean-centering across frequency gives spectral contrast — which frequencies are louder than average.
- **Convolutional SRL layers:** 1D convolution across frequency axis. Templates learn spectral patterns — formants, harmonics, noise bands.
- **Temporal composition is essential:** A single spectral snapshot is near-meaningless. Phonemes are defined by spectral *change* over tens of milliseconds, words over hundreds. The EMA hierarchy naturally separates these timescales.

### Motor Cortex
- **Input:** Proprioceptive state — muscle tensions, joint angles. A vector where each dimension is a sensor reporting local state.
- **Mean-centering:** "How is my current pose deviating from my average pose."
- **Feedback as gain:** The goal arrives as a contrastive concept from higher layers, projected down through the weight chain to become a gain field over the proprioceptive input.
- **Critical inversion:** In visual cortex, the gain modulates what the system *perceives*. In motor cortex, the lowest layer's output connects to *actuators*, not to a representation. The gain that in vision means "attend to this" in motor cortex means "do this." The residual between the gain-modulated proprioceptive input and the current reconstruction is the motor error — what needs to change.
- **Same architecture:** The motor hierarchy translates "reach for the cup" → "extend arm" → "contract these muscles by these amounts" through the same cascading gain mechanism. The only difference from sensory cortex is what the wires at the bottom connect to.

### Somatosensory Cortex
- **Input:** 2D body-surface map (pressure, temperature at each point).
- **Mean-centering:** "What's touching me more than average."
- **Templates:** Learn tactile patterns — grip, edge, texture.
- **Temporal composition:** Captures dynamics like stroking, vibration, holding.

### Olfactory Cortex
- **Input:** Vector of chemical receptor activations.
- **Mean-centering:** "What's chemically distinctive about this smell versus baseline."
- **Minimal spatial structure:** Almost purely a high-dimensional vector without topology to convolve over.
- **Heavily temporal:** Slow EMA timescales.

### Vestibular Cortex
- **Input:** Orientation and acceleration (three rotational axes, three linear acceleration axes).
- **Mean-centering:** "How is my motion deviating from steady state."
- **Feeds into motor cortex** as a reference frame for motor planning.

### Prefrontal Cortex
- **Input:** Top-level activations from all other cortices.
- **Same contrastive tiling** on the manifold of multimodal abstract representations.
- **Templates learn patterns** like "I see food and I'm hungry and my hand is free."
- **Slowest EMA:** Tracks goals, plans, sustained intentions over minutes or longer.
- **Feedback cascades simultaneously** into every other cortex, providing the top-level gain that says "right now, the thing that matters is this."

---

## Reference Frame Binding (Parietal Cortex)

Each sensory surface is its own reference frame — the retina is eye-centered, muscle spindles are in muscle-length coordinates, etc. Useful behavior requires relating them.

The parietal cortex receives top-level activations from multiple sensory streams — visual location, eye proprioception, head proprioception, vestibular orientation. Each is a signed, mean-centered contrastive vector. The parietal layer is just another SRL layer doing contrastive tiling on the concatenation of these inputs.

A parietal template might learn "retinal position X, eye position Y, head position Z" — a specific multimodal configuration that corresponds to a specific world-relative location. Different combinations of retinal position and eye position that correspond to the same world location produce similar downstream effects (reaching to the same place, getting the same reward). Upper layers learn that these parietal templates are equivalent.

The coordinate transform isn't computed explicitly. It's learned as a regularity in the joint activation space. The invariant — world-relative position — emerges as a cluster in the parietal manifold. No matrix multiplication, no trigonometry. Just contrastive tiling on high-dimensional input that happens to contain multiple reference frames.

The body schema itself emerges from compositional contrastive tiling of proprioceptive inputs. "Head orientation" isn't stored as Euler angles — it's the activation pattern across templates that tile the manifold of neck joint configurations.

---

## Simulated Motor Agent as First Testbed

A simulated agent with clean state variables (joint angles, joint velocities) provides a low-dimensional testbed for the full architecture:

- **State space:** Maybe 4-6 dimensions (e.g., a 2D planar arm with two joints — two angles, two velocities).
- **Input:** Proprioceptive state vector. Mean-center, normalize, process with SRL.
- **Feedback:** Target pose arrives as a contrastive gain field.
- **Motor output:** Lowest layer's output drives the joints.
- **Temporal composition:** Reaching requires a trajectory, exercising the EMA and recurrent mechanisms.
- **Closed-loop evaluation:** Either the arm reaches the target or it doesn't. Immediate, concrete, visual.

This exercises every component except spatial convolution: contrastive input processing, SRL tiling, temporal composition, feedback as gain, motor output, and the recurrent loop for sequential movements. The dimensionality is low enough to visualize everything — template manifolds, gain fields, trajectories. Vision can be added later (a camera seeing its own arm), introducing the parietal binding between visual and proprioceptive streams.

---

## Core Thesis

A single operation — **signed, mean-centered contrastive deviation** — is simultaneously:

- The **feature descriptor** (pixel contrast → template activations)
- The **learning signal** (LOO residual → geodesic rotation)
- The **inhibition mechanism** (LOO shadow → competitive suppression)
- The **categorical identity** (class contrast → what it is and isn't)
- The **attentional gain** (projected contrast → multiplicative spotlight)
- The **motor command** (gain-modulated residual → muscle activation)
- The **surprise signal** (reconstruction error → cascading incoherence)
- The **temporal context** (EMA-smoothed contrast → what's been happening)

All of these are the same operation applied at different levels of abstraction to different types of input. The functional diversity of the brain emerges not from specialized circuits but from one universal contrastive circuit connected to different sensors and actuators, running at different timescales.

What a layer does, the network does compositionally. What the network does, the brain does across modalities.
