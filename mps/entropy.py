import numpy as np


def entanglement_entropy_from_singular_values(S, eps=1e-15):
    """
    Compute von Neumann entanglement entropy from singular values S.
    """
    w = np.abs(S) ** 2
    Z = np.sum(w)
    if Z < eps:
        return 0.0

    p = w / Z
    p = p[p > eps]  
    return float(-np.sum(p * np.log(p)))