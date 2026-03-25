# Inhibition and Learning Decoupling

## Notes from a brainstorming session — 2026-03-25

---

## Starting Point

Starting from an iterative inhibition model where stored inhibition is applied across ticks, with reconstruction subtraction and LOO lateral competition.

---

## Iterative Inhibition Across Ticks

The first approach: store inhibition values between process calls. Each tick:

1. Compute raw activations: `a_raw = ReLU(W . x_hat)`
2. Apply stored inhibition: `a_inh = ReLU(a_raw - stored_inhibition)`
3. Output `a_inh`, compute new inhibition for next tick: `inh_i = (sum(a_inh) - a_inh_i) / (k-1) * lambda`

The idea was that with a static input, activations keep inhibiting each other across ticks until converging to a stable pattern. First tick has zero inhibition (raw activations), each subsequent tick refines.

### Convergence Analysis

For two symmetric templates with `a_raw = 0.5`, the fixed point is:

`a = a_raw / (1 + lambda)`

- lambda=0.5: a = 0.333
- lambda=1.0: a = 0.25

The iteration `s_{n+1} = (a_raw - s_n) * lambda` has derivative `-lambda`. Converges only when **lambda < 1**. At lambda=1 it oscillates forever, at lambda>1 it diverges.

---

## Closed-Form Equilibrium

Since the iterative process converges to a known fixed point, we can compute it directly without iteration. With `c = lambda / (k-1)`:

1. Find the active set A where `a_raw_i > c * S_A`
2. `S_A = sum_A(a_raw) / (1 + (|A|-1) * c)`
3. `a_i = (a_raw_i - c * S_A) / (1 - c)` for active templates, 0 otherwise

The active set is found by starting with all positive `a_raw` and shrinking until stable (at most k steps, typically 2-3).

This works for any `lambda < k-1` (e.g., up to 24 with 25 templates), removing the lambda<1 constraint of the iterative approach. No stored state needed across ticks.

### Connection to Partial Correlation

This closed-form LOO inhibition is mathematically equivalent to computing **partial correlations** — each template's correlation with the input after removing the variance explained by all other active templates. Also equivalent to computing multiple regression coefficients.

The neuroscience motivation (lateral inhibition) and the statistics motivation (partial correlation) converge on the same solution because they solve the same problem: given multiple overlapping explanations, how much does each uniquely contribute?

---

## Learning Rule: Residual Chasing

All templates chase the residual `x_hat - recon_hat`, with rotation angle scaled inversely by activation: `theta_i = eta * (1 - a_i)`.

- Winners (a=1): don't move, they already explain the input
- Losers (a=0): rotate at full eta toward what's left unexplained
- If input is fully explained: residual is zero, nobody moves (natural fixed point)

### Reconstruction Normalization

The reconstruction `recon = a_inh @ W` can overshoot `||x_hat||` when multiple correlated templates are active (their weighted contributions add up past magnitude 1). The residual `x_hat - recon` flips sign, causing losers to chase the anti-direction of the input.

L2-normalizing the reconstruction (`recon_hat = recon / ||recon||`) prevents the flip but erases magnitude information — a weak partial explanation looks as confident as a full one, producing misleading residuals.

Clipping to max norm 1 was considered as a middle ground: `recon_clipped = recon * min(1, 1/||recon||)`. Preserves magnitude when weak, prevents flip when overshooting.

This remains an open issue.

---

## The Decoupling Insight

### Inhibition Changes the Semantics of Activations

Raw activations are honest Pearson correlations — a template's activation reflects its geometric similarity to the input. This is a fact about the template-input relationship that doesn't change because other templates exist.

Inhibition reduces activations based on what others are doing. A vertical line template with 0.5 correlation to a plus sign gets pushed to 0.25 because a horizontal line template also responded. But why should the vertical template's confidence change because of an unrelated hypothesis?

### Partial vs Raw Correlations

For non-orthogonal templates, the two representations differ after mean-centering and L2-normalization:

- **Raw correlations**: distributed representation. "Several things match." Preserves fine-grained differences among weak activations.
- **Partial correlations**: sharpened representation. Amplifies the ratio between dominant and secondary matches. Weak/redundant activations are suppressed or zeroed out.

Example with a bright-vertical plus sign (vertical=0.7, horizontal=0.5, diagonal=0.1):
- Raw ratio (v:h) = 1.4x
- Partial ratio (v:h) = 2.0x (diagonal zeroed out entirely)

### The Orthogonality Argument

If templates converge to be truly distinct (the learning goal), raw and partial correlations become equivalent after normalization — inhibition scales all activations uniformly when there's no redundancy. The sharpening only matters when templates overlap.

This suggests: rather than fixing the output with inhibition, let learning make the templates distinct enough that sharpening isn't needed.

### The Perceptual Argument

When you see a plus sign, you perceive "vertical and horizontal" — not "also a bit diagonal." Perception is already sharpened. But this might be because the brain's templates are already distinct enough that the diagonal doesn't fire, not because inhibition suppresses it after the fact.

---

## Key Conclusion: Inhibition Serves Learning, Not Output

In biological systems, activations and learning are tightly coupled. But conceptually, the purpose of lateral inhibition may be primarily to **drive learning dynamics** (who specializes on what), with the activation pattern being a side effect.

The raw Pearson correlation is an honest, geometrically grounded output. The inhibited activation is a competition-adjusted signal useful for deciding who learns from what.

These two concerns may be separable:
- **Output to next layer**: raw correlations, mean-centered and L2-normalized (honest, preserves structure)
- **Learning signal**: inhibited activations determine who chases the residual and how hard

Whether to actually decouple them or keep them coupled (as biology does) remains an open design decision. The current implementation keeps them coupled — the closed-form equilibrium activations are both the output and the learning driver.

---

## Open Questions

- **Should output and learning activations be decoupled?** Output raw correlations but use inhibited activations internally for learning dynamics.
- **Reconstruction normalization**: L2-normalize vs clip vs raw — what's the right treatment to prevent residual sign flips without losing magnitude information?
- **Does the next layer benefit more from independent components (partial) or full structure (raw)?** May depend on the task: classification favors sharpening, compositional representation favors distributed.
- **Optimal lambda**: governs how many templates survive competition. With closed-form, any lambda < k-1 works. But what value produces the best template specialization?
- **Energy conservation**: sum of squared raw activations <= 1 for orthogonal templates (Bessel's inequality). Could this be used as a diagnostic for template redundancy? If sum(a_i^2) > 1, templates are explaining overlapping variance.
