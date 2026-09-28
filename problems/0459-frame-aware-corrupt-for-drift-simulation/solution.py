import numpy as np
from scipy.ndimage import uniform_filter

def frame_aware_corrupt(
    frames: np.ndarray,
    gaussian_prob: float,
    gaussian_std: float,
    color_shift_prob: float,
    color_shift_range: float,
    blur_prob: float,
    blur_kernel_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply per-frame stochastic corruption to simulate history drift.

    Args:
        frames:            float array of shape (T, H, W, C)
        gaussian_prob:     probability of Gaussian noise per frame
        gaussian_std:      std of Gaussian noise
        color_shift_prob:  probability of color shift per frame
        color_shift_range: uniform shift range per channel
        blur_prob:         probability of mean blur per frame
        blur_kernel_size:  square kernel side length for blur
        rng:               seeded numpy random generator

    Returns:
        Corrupted array of shape (T, H, W, C), dtype float
    """
    frames = frames.astype(np.float32)
    
    # Loop over each frame independently
    for t in range(frames.shape[0]):
        # Gaussian noise
        if rng.random() < gaussian_prob:
            noise = rng.normal(0, gaussian_std, frames[t].shape).astype(np.float32)
            frames[t] = frames[t] + noise
        
        # Color shift
        if rng.random() < color_shift_prob:
            shift = rng.uniform(-color_shift_range, color_shift_range, frames.shape[-1])
            frames[t] = frames[t] + shift
        
        # Blur
        if rng.random() < blur_prob:
            blurred = np.zeros_like(frames[t])
            for c in range(frames.shape[-1]):
                blurred[:, :, c] = uniform_filter(
                    frames[t, :, :, c],
                    size=blur_kernel_size,
                    mode='nearest'
                )
            frames[t] = blurred
    
    return frames




