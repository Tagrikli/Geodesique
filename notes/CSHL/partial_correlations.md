# CSHL: From Iterative Inhibition to Direct Partial Correlations

## What changed

The previous CSHL implementation used an iterative inhibition loop that ran one step per tick:

```
a_raw = ReLU(W @ x_hat)
recon = a @ W
inhib = W @ recon
a = ReLU(a_raw - (inhib - a))
```

This was a projected gradient descent for NNLS, converging over multiple ticks.
It also used ReLU, discarding negative correlations.

We replaced it with a direct partial correlation computation:

```
c = W @ x_hat                    # Pearson correlations
G_inv = inv(W @ W^T)             # inverse Gram matrix
beta = G_inv @ c                 # regression coefficients
a_i = beta_i / sqrt(G_inv_{ii})  # partial correlations
```

## Why

1. **Negative correlations are signal.** Each template is a hypothesis about the input. Anti-correlation means the input is structurally opposed to that hypothesis — stripping that out with ReLU was throwing away information.

2. **Iterative convergence was unnecessary.** The iteration solved `G a = c` one step at a time across ticks, meaning activations lagged behind the true solution. With k=25 templates, a direct `inv(G)` is trivial (25x25 matrix) and gives the exact answer every tick.

3. **Partial correlations are the right quantity.** Raw Pearson correlations conflate a template's unique contribution with variance shared across correlated templates. Partial correlations factor that out — each activation reflects only the unique explanatory power of that template given all the others.

## Learning rule

The learning rule remained the same: geodesic rotation on the unit hypersphere.

```
theta_i = eta * a_i
w_i' = w_i * cos(theta_i) + tau_hat_i * sin(theta_i)
```

where `tau_hat_i` is the unit tangent from `w_i` toward `x_hat`. Positive partial correlation rotates toward the input, negative rotates away.

The geodesic rotation preserves both unit norm and mean-centering exactly (since `tau` is a linear combination of zero-mean vectors `x_hat` and `w`), so the re-centering/re-normalization step at the end is technically redundant — kept only as a floating-point drift guard.

## Open concern

The direct partial correlation computation requires a matrix inverse every tick. For k=25 this is fine, but the O(k^3) cost and the conceptual weight of `np.linalg.inv` is bothersome. There may be a cleaner formulation that avoids the explicit inverse — possibly an iterative scheme that converges in activation-space without needing to materialize G^{-1}, or a decomposition that makes the per-tick cost lighter and more interpretable.
