import numpy as np


def svd_truncate(M, chi_max):
    """
    SVD factorization with truncation.

    M: 2D numpy array (matrix)
    chi_max: max number of singular values to keep

    Returns:
      U, S, Vh, trunc_error

    trunc_error is the sum of discarded singular values squared
    divided by total sum of singular values squared (a relative error measure).
    """
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    # compute truncation error
    total = np.sum(S**2)
    if chi_max is None or chi_max >= len(S):
        trunc_error = 0.0
        return U, S, Vh, trunc_error

    kept = S[:chi_max]
    discarded = S[chi_max:]

    disc = np.sum(discarded**2)
    trunc_error = disc / total if total > 0 else 0.0

    U = U[:, :chi_max]
    S = kept
    Vh = Vh[:chi_max, :]

    return U, S, Vh, trunc_error