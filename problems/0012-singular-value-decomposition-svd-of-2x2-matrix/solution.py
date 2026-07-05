import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Let A = U @ S @ V.T
    
    ## Compute V
    # Find A transpore
    A_tA = A.T @ A
    eigenvalues, V = np.linalg.eigh(A_tA)
    eigenvalues = np.flip(eigenvalues)
    V = np.flip(V, axis=1)
    ## Compute S
    S = np.sqrt(eigenvalues)
    return (V, S, V.T)