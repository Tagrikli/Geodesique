# SRL vs CRG and Channel-Based Convolutional Processing

## SRL vs CRG

CRG uses unsigned activations with ReLU gating --- only positively activated templates learn and contribute to reconstruction. SRL removes the ReLU entirely. All templates participate with their signed cosine similarity. The sign flip in the residual (`sign(s_i) * (x_hat - x_shadow_hat_i)`) makes anticorrelated templates organize the opposite hemisphere, positioning themselves so their negative activation is informative. One learning rule, no active gate, full symmetric learning.

CRG required a balance parameter for E/I forces. SRL has none --- the LOO residual encodes both attraction and repulsion inherently. SRL also drops the sparsity parameter; sparsity emerges from LOO shadow inhibition.

---

## Spatial Tiling to Channel-Based Convolution

### The Problem with Spatial Tiling

The original ConvSRL tiled template activations spatially: for K templates per patch on a grid of patches, each patch's K activations were arranged as a sqrt(K) x sqrt(K) block, producing a 2D mosaic output. This required:

- Template count must be a perfect square (4, 9, 16, 25...)
- Next layer's kernel size must be a multiple of sqrt(K) to avoid splitting hypercolumn boundaries
- Architecture is locked into rigid size chains (e.g., 28x28 -> 70x70 -> 35x35 -> 5x5 with only 3 layers possible)
- Same-sized outputs between layers means no spatial compression, no hierarchy

The advantage was that every layer's output was a directly viewable 2D image, making reconstruction trivially visual at every layer. (Silly me haha)

### Channel-Based Approach

ConvSRL now outputs a 3D tensor `(grid_h, grid_w, K)` where K is the channel/template dimension. The next layer receives this 3D tensor and tiles it into patches of shape `(P, P, C)` which flatten to `P*P*C`-dimensional vectors for learning on the hypersphere.

This removes all constraints:
- Template count can be any integer
- Kernel size only needs to divide the spatial dimensions
- Layers compose freely

The input handling distinguishes 2D inputs (first layer from ConvLGN) from 3D inputs (deeper layers from previous ConvSRL). For 2D: `dim = P*P`. For 3D with C channels: `dim = P*P*C`. Weights auto-initialize to the correct dimensionality.

### Reconstruction with Channels

ConvReconstruction infers channel count from weights: `C = dim // (P*P)`. If C=1, output is 2D. If C>1, output is 3D. Multi-layer reconstruction chains backward: layer N reconstructs to `(H, W, C)` which feeds into layer N-1's reconstruction.

Middle layer reconstructions are no longer directly viewable as images --- they are high-dimensional channel tensors. This is the tradeoff for architectural freedom.

### Connection to Standard CNNs

This is how conventional CNNs work: a 3x3 kernel on RGB input operates on 3x3x3 = 27 values. Channels are just more dimensions in the flattened patch vector. The flattening is the key insight --- the hypersphere doesn't care whether the 100 dimensions came from a 10x10 spatial tile or a 5x5x4 spatial+channel volume. Templates learn to find structure across all of it.
