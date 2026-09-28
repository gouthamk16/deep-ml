import numpy as np

def rmsNorm(x):
    x = np.array(x, dtype=np.float32)
    weights = np.ones(x.shape[-1], dtype=np.float32)
    epsilon = 1e-6
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + epsilon)
    norm = (x / rms) * weights
    return norm

def normalize(x):
    '''
    Apply normalization to a batch of feature vectors.
    
    Args:
        x: numpy array of shape (batch_size, features)
           Raw activations from a neural network layer
    
    Returns:
        numpy array of same shape (batch_size, features)
        Normalized activations
    
    Requirements:
        - Must not be the identity function
        - Must work on 2D arrays of any size
        - Must be deterministic
        - Must not produce NaN or Inf
        - Normalize across the LAST dimension (features)
    
    Hint: Modern LLMs use RMSNorm, which normalizes by the
    root mean square of the feature values. But you can
    implement any normalization you like!
    '''
    # TODO: Implement your normalization function
    
    result = rmsNorm(x)  # Replace with your normalization
    
    return result
