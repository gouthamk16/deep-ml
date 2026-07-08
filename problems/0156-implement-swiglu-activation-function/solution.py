import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Assuming beta = 1
    swiglu = np.zeros((x.shape[0], x.shape[1]//2))
    for i in range(x.shape[0]):
        x1, x2 = x[i, 0:x.shape[1]//2], x[i, x.shape[1]//2:x.shape[1]]
        print(x1, x2)
        sigmoid_x2 = 1 / (1 + np.exp(-1*x2))
        swish = x2 * sigmoid_x2
        swiglu[i] = x1 * swish

    return swiglu 