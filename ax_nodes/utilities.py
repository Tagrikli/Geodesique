import numpy as np

def to_display_grid(arr, patch_shape=None):
    """Convert 1D or 2D array to a tiled 2D mosaic for display.

    Args:
        arr: 1D vector or 2D matrix (M rows of D-length vectors).
        patch_shape: (H, W) tuple describing how each D-length vector should
                     be reshaped for display.  If None, assumes sqrt(D) square.

    For 1D arrays: reshape using patch_shape or nearest square.
    For 2D arrays (e.g. weight matrix): tile each row as an (H, W) patch.
    """
    import numpy as np

    if arr.ndim == 1:
        n = len(arr)
        if patch_shape is not None:
            H, W = patch_shape
            padded = np.zeros(H * W, dtype=arr.dtype)
            padded[:n] = arr
            return padded.reshape(H, W)
        h = int(np.ceil(np.sqrt(n)))
        w = int(np.ceil(n / h))
        padded = np.zeros(h * w, dtype=arr.dtype)
        padded[:n] = arr
        return padded.reshape(h, w)
    elif arr.ndim == 2:
        M, D = arr.shape

        if patch_shape is not None:
            H, W = patch_shape
        else:
            sqrt_D = int(np.sqrt(D))
            if sqrt_D * sqrt_D == D:
                H, W = sqrt_D, sqrt_D
            else:
                W = int(np.sqrt(D))
                while W > 0 and D % W != 0:
                    W -= 1
                if W == 0:
                    W = 1
                H = D // W

        S = int(np.ceil(np.sqrt(M)))
        total = S * S
        pad = total - M

        weights = arr
        if pad > 0:
            weights = np.vstack([weights, np.zeros((pad, D), dtype=weights.dtype)])

        mosaic = (
            weights.reshape(S, S, H, W).transpose(0, 2, 1, 3).reshape(S * H, S * W)
        )
        return mosaic
    else:
        return arr

def scale_to_bwr(arr):
    """Scale array values to [-1, 1] range for full bwr color utilization.

    Uses min-max normalization: values are linearly mapped from [min, max]
    to [-1, 1]. This ensures unit vectors and other normalized data use
    the full blue-white-red color range instead of appearing whitish.
    """
    arr = np.asarray(arr)
    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        # Constant array - return zeros to show as white/neutral
        return np.zeros_like(arr)

    # Scale to [-1, 1]: 2 * (x - min) / (max - min) - 1
    return 2 * (arr - min_val) / (max_val - min_val) - 1

def softmax(arr, axis=-1, temperature=1.0):
    """Numerically stable softmax with temperature scaling."""
    arr = np.asarray(arr)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    arr = arr / float(temperature)
    max_val = np.max(arr, axis=axis, keepdims=True)
    shifted = arr - max_val
    exp = np.exp(shifted)
    sum_exp = np.sum(exp, axis=axis, keepdims=True)
    return exp / sum_exp
