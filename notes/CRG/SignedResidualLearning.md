# Signed Residual Learning

We had already enabled negative activations for the output — the representation was signed. But the learning rule wasn't. Previous CRG implementations used unsigned residuals: `r_i = x_hat - x_shadow_hat_i` for all templates, and only positively activated templates learned. The sign multiplication in the residual is the missing piece.

## What the sign flip does

### Positively correlated templates (s_i > 0)

The residual `x_hat - x_shadow_hat_i` pushes toward unclaimed input structure and away from what the collective already explains. The template moves toward whatever the collective is missing. This is the standard CRG behavior — nothing changes here.

### Anticorrelated templates (s_i < 0)

The flipped residual `x_shadow_hat_i - x_hat` does something really interesting. It pushes the template toward what the collective over-explains and away from the raw input. The template is moving toward the shadow and away from the data.

At first that sounds wrong — why would you want a template to move away from the data? But remember the signed coding framework. This template's job isn't to match the input. Its job is to be in a position where its negative activation is informative. By moving toward the shadow (what others explain) and away from the input, it's positioning itself to be maximally anticorrelated with the residual structure — the stuff that's hard to explain. It's finding the position where its negative deviation says "I am specifically not this."

## The symmetry is geometric

Think about the sphere. The positively correlated templates are on the same side as the input. They're competing to explain what's there. The negatively correlated templates are on the opposite side. By flipping the residual, you're telling them to compete to explain what's not there — or more precisely, to position themselves so that their negative response is maximally structured.

The two groups of templates are doing the same thing in their respective hemispheres. Positive templates organize around the input. Negative templates organize around the anti-input direction. And the LOO mechanism ensures that within each hemisphere, templates don't redundantly overlap.

This is genuinely symmetric. One learning rule, one residual, one sign flip. No branches, no special cases.

## Implementation

```
a_i = s_i = w_i . x_hat                           (signed, no ReLU)
x_recon = sum_j s_j * w_j                          (full signed reconstruction)
x_shadow_i = x_recon - s_i * w_i                   (LOO shadow)
r_i = sign(s_i) * (x_hat - x_shadow_hat_i)         (sign-aware residual)
```

Then standard Rodrigues rotation from the residual's tangent projection. All templates learn — no active gate needed.
