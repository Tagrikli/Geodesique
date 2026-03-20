# Category Alignment and the Role of Dopamine

## The Problem

SRL templates tile the input manifold based on geometric structure, not task utility. Two structurally similar but categorically distinct inputs (sneaker vs ankle boot, horse vs dog) produce similar activations. A linear classifier on top struggles to separate them because the representation wasn't shaped by the classification objective.

Backprop CNNs solve this by optimizing features end-to-end for the task. But their features aren't manifold-aligned --- they encode whatever minimizes the loss, even if the corresponding weight directions don't point to regions where real data lives. SRL templates stay on the data manifold; backprop features don't have to.

---

## Supervised Rotation is Riemannian SGD

If we compute the rotation that would produce a desired activation pattern, we get the tangent vector from the loss gradient projected onto the tangent plane at each template's position on the sphere. This is Riemannian SGD on the unit hypersphere (Absil, Mahony & Sepulchre, 2008). With multiple layers, computing these tangent vectors requires backpropagating the error through each layer --- this collapses to standard backprop with a spherical weight constraint.

The unsupervised LOO tangent and the supervised loss tangent are both tangent vectors on the same manifold. They can be blended:

```
tau_final = (1 - alpha) * tau_unsupervised + alpha * tau_supervised
```

At alpha=0: pure self-organization. At alpha=1: Riemannian SGD (effectively backprop on the sphere). The middle ground is where something genuinely new lives.

### The Manifold Problem with Supervision

A supervised signal wants to rotate templates to produce target activations, but the required weight positions may not correspond to regions of the hypersphere where data actually exists. The supervised tangent can pull templates off the manifold into empty space. The unsupervised component is structurally necessary to keep templates grounded. Pure supervision breaks the geometric interpretation.

---

## Dopamine as Reinforcement

Rather than prescribing activations (supervised) or computing gradients (backprop), a dopamine-like signal says "good" or "bad" without specifying direction. The templates figure out how to get more reward within their own geometric dynamics.

### The Direction Problem

A pure scalar "good/bad" signal carries no directional information. Templates have no way to know which way to move to get more reward. They'd randomly drift and occasionally get reinforced --- incredibly slow. Any useful dopamine mechanism must either carry implicit direction or create conditions where the existing learning rule naturally finds the right direction.

### Dopamine Doesn't Modulate Learning Rate

Initial intuition: reward scales eta per template. But if a template is already doing well, speeding it up makes it overshoot. And the LOO residual points the same direction regardless of reward --- more learning from the same signal doesn't help discrimination because structurally similar inputs (horse/dog) produce similar residuals.

---

## The Dog-Horse Problem

A child sees a horse and says "dog." The parent doesn't give sugar. The child needs to learn to discriminate without being told the answer.

### Approach 1: Look Here (Amplify Differences)

Compare the input against a confusable instance from another class. Compute `gain = |X - Y|`. Apply gain to suppress shared structure (body, legs) and amplify discriminating structure (face, ears, size). Templates learn from the gain-modulated input.

**Problem:** Within-class variance. Two very different-looking dogs produce high gain everywhere --- the mechanism assumes same-class instances are structurally similar, which isn't always true (open 4 vs closed 4).

### Approach 2: Compare Against Prototype

Instead of comparing against a specific instance, compare against the class prototype (centroid/concept). The child compares the horse to its *idea* of a dog, not a specific dog it saw.

- Horse input vs dog prototype: highlights "what makes this not-a-dog" --- useful for discrimination
- Dog input vs dog prototype: highlights "what's unusual about this dog" --- within-class variance, not discrimination

The direction matters: only compare against the prototype of the class you're *mistaking it for*.

### Approach 3: Don't Look There (Suppress What Causes Confusion)

When two classes are confused, identify what's shared (the templates that fire similarly for both). Suppress those contributions in the input. The templates can no longer rely on shared structure and must find other features that still reconstruct the input but happen to be more discriminative.

No target, no direction --- just "this isn't enough, find something else." Pure dissatisfaction signal.

**Limitation:** This separates confused categories but doesn't unify within-class representations. A weird-looking 4 and a normal 4 might develop very different activations even though they're the same category.

---

## The Invariance-Equivariance Tension

Two contradictory requirements from a single representation:

- **Invariance:** All dogs produce similar activations (categorization)
- **Equivariance:** Each dog retains its unique visual details (reconstruction)

If activations are invariant to within-class differences, reconstruction detail is lost. If they preserve all details, categorization fails.

These may need to be **separate streams** propagated to upper layers:

1. **Reconstruction stream:** Full activation vector preserving visual detail (what SRL naturally produces)
2. **Category stream:** Compressed signal that maps different-looking instances of the same class to similar representations

Analogous to ventral ("what") and dorsal ("where") streams in visual cortex, but here: "what does it look like" vs "what category is it."

---

## The Learning Signal

```
learning signal = unexplained error * what it is * what it is not
```

The three factors:
- **Unexplained error:** What the current representation fails to capture (the LOO residual --- this is the unsupervised component)
- **What it is:** The identity/structure of the current input (keeps templates on the manifold)
- **What it is not:** The contrastive/discriminative signal (what confusable categories look like that this input doesn't)

All three are necessary. Without the unexplained error, there's no learning signal. Without "what it is," templates drift off the manifold. Without "what it is not," there's no discrimination.
